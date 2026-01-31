import streamlit as st
import pandas as pd
import json
import re
import plotly.express as px
from datetime import datetime
from google import genai
from google.genai import types
import firebase_admin
from firebase_admin import credentials, firestore

# --- 1. Firebase & 설정 (Cloud DB) ---

def init_firebase():
    """Firestore 연결을 초기화합니다."""
    if not firebase_admin._apps:
        try:
            sa_info = json.loads(st.secrets["FIREBASE_SERVICE_ACCOUNT"])
            cred = credentials.Certificate(sa_info)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase 초기화 실패: {e}. 'FIREBASE_SERVICE_ACCOUNT' 시크릿을 확인하세요.")
            return None
    return firestore.client()

db = init_firebase()
APP_ID = "Good_Morning_v1"

ADMIN_PASSWORD = "1111" 
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025" 
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. Firestore 데이터 조작 함수 ---

def get_players():
    """Firestore에서 도토리 명단을 실시간으로 가져옵니다."""
    if not db: return pd.DataFrame(columns=['name', 'elo'])
    players_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'players')
    docs = players_ref.stream()
    player_list = [doc.to_dict() for doc in docs]
    return pd.DataFrame(player_list) if player_list else pd.DataFrame(columns=['name', 'elo'])

def get_matches():
    """Firestore에서 전체 경기 이력을 가져옵니다."""
    if not db: return pd.DataFrame()
    matches_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'matches')
    docs = matches_ref.stream()
    match_list = []
    for doc in docs:
        d = doc.to_dict()
        d['id'] = doc.id
        match_list.append(d)
    return pd.DataFrame(match_list) if match_list else pd.DataFrame()

def get_round_start_elos(round_name):
    """특정 차수의 경기를 계산하기 위해, 해당 차수 시작 시점의 ELO 스냅샷을 가져옵니다."""
    if not db: return {}
    df_p = get_players()
    start_elos = {name: 1500.0 for name in df_p['name']}
    
    # 해당 차수보다 숫자가 낮은 차수 중 가장 마지막 기록을 찾음
    target_val = extract_round_number(round_name)
    
    # Firestore 쿼리: 현재 입력하려는 차수보다 이전 데이터 중 최신 1개
    # (참고: 복합 색인이 필요할 수 있으나, 단일 필드 쿼리 후 파이썬에서 필터링하는 방식으로 안전하게 구현)
    for name in df_p['name']:
        p_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(name)
        history_query = p_ref.collection('history').where('round_val', '<', target_val).order_by('round_val', direction=firestore.Query.DESCENDING).limit(1).get()
        
        if history_query:
            start_elos[name] = history_query[0].to_dict().get('elo', 1500.0)
            
    return start_elos

def get_next_round_name():
    """마지막 차수를 확인하여 다음 차수 이름을 생성 (예: 10차 -> 11차)"""
    df_m = get_matches()
    if df_m.empty:
        return "1차 정기전"
    
    # 가장 높은 round_val 찾기
    df_m['round_val'] = df_m['round'].apply(extract_round_number)
    last_round_num = int(df_m['round_val'].max())
    next_round_num = last_round_num + 1
    
    # 마지막 차수의 이름을 참고하여 숫자만 변경 (예: "10차 정기전" -> "11차 정기전")
    last_round_name = df_m.loc[df_m['round_val'].idxmax(), 'round']
    next_round_name = re.sub(r'\d+', str(next_round_num), last_round_name)
    
    return next_round_name

def save_match_to_cloud(match_data, w_avg, l_avg):
    """경기 결과 저장 및 해당 시점의 ELO 히스토리 즉시 기록"""
    if not db: return
    
    # 경기 데이터에 당시 계산 기준 점수 추가
    match_data['w_avg_at_match'] = w_avg
    match_data['l_avg_at_match'] = l_avg

    # 1. 경기 데이터 저장
    matches_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'matches')
    _, match_doc_ref = matches_ref.add(match_data)
    
    change = match_data['elo_change']
    round_name = match_data['round']
    round_val = extract_round_number(round_name)
    
    winners = [match_data['winner1'], match_data['winner2']]
    losers = [match_data['loser1'], match_data['loser2']]
    
    # 2. 관련 선수들 점수 업데이트 및 히스토리 추가
    for p_name in winners + losers:
        is_winner = p_name in winners
        actual_change = change if is_winner else -change
        
        p_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(p_name)
        
        # Firestore 트랜잭션 대신 간단하게 Incremnet 사용
        p_ref.update({"elo": firestore.Increment(actual_change)})
        
        # 업데이트된 최종 점수를 가져와서 히스토리에 기록
        updated_elo = p_ref.get().to_dict().get('elo', 1500.0)
        
        # history 서브 컬렉션에 추가 (그래프 자동 업데이트용)
        p_ref.collection('history').add({
            "elo": updated_elo,
            "change": actual_change,
            "round": round_name,
            "round_val": round_val,
            "timestamp": match_data['timestamp'],
            "match_id": match_doc_ref.id
        })

def add_new_player(name):
    """새 도토리 등록"""
    if not db: return False
    p_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(name)
    if not p_ref.get().exists:
        p_ref.set({"name": name, "elo": 1500.0})
        return True
    return False

# --- 3. 핵심 분석 및 무승부 개선 로직 ---

def extract_round_number(round_str):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(round_str))
    return float(numbers[0]) if numbers else 0.0

def calculate_elo_logic(w_avg, l_avg, score_text, k=32):
    """
    ELO 변동폭 계산 로직 (무승부 합리적 개선 포함).
    w_avg: A팀(Winner 표기팀) 평균 / l_avg: B팀(Loser 표기팀) 평균
    """
    # A팀의 기대 승률 (0.5면 동률)
    expect_win = 1 / (10 ** ((l_avg - w_avg) / 400) + 1)
    
    is_draw = False
    score_diff = 1
    try:
        if ':' in score_text:
            s1, s2 = map(int, score_text.split(':'))
            score_diff = abs(s1 - s2)
            if s1 == s2: is_draw = True
    except: pass

    if is_draw:
        # 무승부 시: 실제 결과값(Actual)을 0.5로 설정
        # 기대승률이 0.4(약팀)라면, 0.5 - 0.4 = +0.1 보너스
        # 기대승률이 0.6(강팀)이라면, 0.5 - 0.6 = -0.1 페널티
        # 기대승률이 0.5(동률)라면, 0.5 - 0.5 = 0
        change = k * (0.5 - expect_win)
    else:
        # 일반 승패 시: 가중치 적용
        if score_diff >= 6: multiplier = 1.5
        elif score_diff >= 4: multiplier = 1.25
        elif score_diff >= 1: multiplier = 1.0
        else: multiplier = 0.8
        
        # A팀이 이겼다는 가정하에 계산 (Actual = 1)
        change = k * (1 - expect_win) * multiplier
        
    return round(expect_win, 4), round(change, 2)

def recalculate_all_cloud_data():
    """차수별 고정 점수 기반 재정산 (1차수 모든 경기 평균 ELO는 1500점 고정)"""
    if not db: return
    df_p = get_players()
    df_m = get_matches()
    if df_m.empty: return

    # 1. 초기화: 히스토리 삭제 및 모든 선수 점수 1500점 설정
    for name in df_p['name']:
        h_refs = db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(name).collection('history').stream()
        for h_doc in h_refs: h_doc.reference.delete()
    
    current_elos = {name: 1500.0 for name in df_p['name']}
    
    # 2. 차수별 정렬
    df_m['round_val'] = df_m['round'].apply(extract_round_number)
    df_m = df_m.sort_values(by=['round_val', 'timestamp']).reset_index(drop=True)
    
    # 3. 차수별 그룹화 처리
    for round_name, group in df_m.groupby('round', sort=False):
        # ⭐ 중요: 차수 시작 시점의 점수를 스냅샷으로 고정
        start_of_round_elos = current_elos.copy()
        round_accumulated_changes = {name: 0.0 for name in df_p['name']}
        last_ts = ""

        for _, row in group.iterrows():
            w1, w2, l1, l2 = row['winner1'], row['winner2'], row['loser1'], row['loser2']
            
            # 차수 시작 시점의 고정 점수로 평균 계산 (1차수라면 모두 1500점)
            w_avg = (start_of_round_elos.get(w1, 1500) + start_of_round_elos.get(w2, 1500)) / 2
            l_avg = (start_of_round_elos.get(l1, 1500) + start_of_round_elos.get(l2, 1500)) / 2
            
            exp, diff = calculate_elo_logic(w_avg, l_avg, row['score'])
            last_ts = row['timestamp']
            
            # 경기 문서 업데이트 (당시 기대승률과 변동폭 저장)
            db.collection('artifacts', APP_ID, 'public', 'data', 'matches').document(row['id']).update({
                "elo_change": diff, 
                "expected_win": exp,
                "w_avg_at_match": w_avg, # 이 값이 저장되어야 히스토리에 정확히 나옵니다.
                "l_avg_at_match": l_avg
            })
            
            # 변동폭 누적 (차수 종료 후 반영하기 위함)
            for p in [w1, w2]: round_accumulated_changes[p] += diff
            for p in [l1, l2]: round_accumulated_changes[p] -= diff

        # 차수 종료 후: 누적된 변동폭을 실제 점수에 반영하고 히스토리 기록
        r_val = extract_round_number(round_name)
        for name in current_elos.keys():
            if round_accumulated_changes[name] != 0:
                current_elos[name] += round_accumulated_changes[name]
                p_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(name)
                p_ref.collection('history').add({
                    "elo": current_elos[name],
                    "change": round_accumulated_changes[name],
                    "round": round_name,
                    "round_val": r_val,
                    "timestamp": last_ts
                })

    # 4. 최종 점수 반영
    for name, final_elo in current_elos.items():
        db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(name).update({"elo": final_elo})

def get_ranking_statistics():
    """상세 랭킹 데이터 산출"""
    df_p = get_players()
    df_m = get_matches()
    if df_p.empty: return pd.DataFrame()
    
    # 전체 진행된 총 차수 구하기
    total_rounds_count = df_m['round'].nunique() if not df_m.empty else 0 #

    stats = {n: {'이름': n, 'ELO 점수': e, '승': 0, '무': 0, '패': 0, '득': 0, '실': 0, '참여차수': set()} 
             for n, e in zip(df_p['name'], df_p['elo'])}
    
    if not df_m.empty:
        for _, row in df_m.iterrows():
            w1, w2, l1, l2 = row['winner1'], row['winner2'], row['loser1'], row['loser2']
            rnd = row['round']

            # 참여한 차수 기록 (중복 제거를 위해 set 사용)
            for p in [w1, w2, l1, l2]:
                if p in stats: stats[p]['참여차수'].add(rnd) #


            try:
                s_parts = row['score'].split(':')
                s_win, s_loss = int(s_parts[0]), int(s_parts[1])
                
                if s_win > s_loss:
                    for w in [w1, w2]: 
                        if w in stats: stats[w]['승'] += 1
                    for l in [l1, l2]: 
                        if l in stats: stats[l]['패'] += 1
                elif s_win == s_loss:
                    for p in [w1, w2, l1, l2]:
                        if p in stats: stats[p]['무'] += 1
                
                for w in [w1, w2]:
                    if w in stats: stats[w]['득'] += s_win; stats[w]['실'] += s_loss
                for l in [l1, l2]:
                    if l in stats: stats[l]['득'] += s_loss; stats[l]['실'] += s_win
            except: pass

    res = []
    for n, s in stats.items():
        played_rounds = len(s['참여차수']) #
        attendance = (played_rounds / total_rounds_count * 100) if total_rounds_count > 0 else 0
        
        # 경기수 계산
        total = s['승'] + s['무'] + s['패']
        wr = (s['승'] / total * 100) if total > 0 else 0
        res.append({
            '이름': n, 
            'ELO 점수': int(round(s['ELO 점수'])),
            '경기수': total, # ⭐ '경기수' 컬럼을 명시적으로 추가
            '승': s['승'], '무': s['무'], '패': s['패'], 
            '득실': s['득'] - s['실'], '승률': int(round(wr)),
            '출석률': int(round(attendance))
        })
    
    df = pd.DataFrame(res)
    if not df.empty:
        # ⭐ 정렬 로직 수정: 
        # 1. 경기수 > 0 여부를 판단하는 임시 컬럼 'has_played' 생성
        df['has_played'] = df['경기수'] > 0
        
        # 2. 정렬 순서: 뛴 사람 먼저 -> ELO 순 -> 승 순 -> 무 순 -> 득실 순
        df = df.sort_values(
            ['has_played', 'ELO 점수', '승', '무', '득실', '이름'], 
            ascending=[False, False, False, False, False, True]
        ).reset_index(drop=True)
        
        # 3. 임시 컬럼 삭제 및 순위 부여
        df = df.drop(columns=['has_played'])
        df.insert(0, '순위', range(1, len(df) + 1))
        
    return df

def analyze_image_with_ai(image_bytes):
    """Gemini API를 사용하여 이미지에서 경기 결과 추출"""
    if not API_KEY:
        st.error("Gemini API Key가 설정되지 않았습니다.")
        return None
    
    client = genai.Client(api_key=API_KEY)
    prompt = """
    이 이미지는 테니스 경기 결과가 적힌 보드입니다. 
    각 경기에서 '승자 2명(winner1, winner2)', '패자 2명(loser1, loser2)', '점수(score)'를 추출하여 JSON 형식으로 응답하세요.
    반드시 다음 구조의 JSON 리스트로 응답해야 합니다:
    {"match_list": [{"winner1": "이름", "winner2": "이름", "loser1": "이름", "loser2": "이름", "score": "6:x"}]}
    이름에 성이 빠져있다면 보이는 대로 적으세요.
    """
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
        )
        # JSON 문자열만 추출하기 위한 정규식 처리
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        st.error(f"AI 분석 중 오류 발생: {e}")
        return None

def add_new_player(name):
    """신규 도토리를 Firestore에 기본 점수로 등록"""
    if not db: return
    p_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(name)
    if not p_ref.get().exists:
        p_ref.set({"name": name, "elo": 1500.0})
        return True
    return False

def display_player_trend(player_name):
    """모든 차수를 표시하고 Y축 범위를 고정한 실력 추이 그래프"""
    if not db: return
    
    # 1. 모든 차수 목록 가져오기 (X축 고정용)
    df_all_matches = get_matches()
    if df_all_matches.empty:
        st.info("기록된 경기 데이터가 없습니다.")
        return
        
    all_rounds = sorted(df_all_matches['round'].unique(), key=extract_round_number)
    
    # 2. 해당 선수의 히스토리 가져오기
    p_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(player_name)
    history_docs = p_ref.collection('history').order_by("timestamp").stream()
    
    history_data = []
    for doc in history_docs:
        history_data.append(doc.to_dict())
    
    # 3. 데이터 재구성: 참여하지 않은 차수도 점수 유지 로직 적용
    plot_data = []
    current_elo = 1500.0  # 초기값
    
    # 히스토리를 차수별 딕셔너리로 변환 (해당 차수의 마지막 점수)
    history_dict = {d['round']: d['elo'] for d in history_data}
    
    for rnd in all_rounds:
        if rnd in history_dict:
            current_elo = history_dict[rnd]
        
        plot_data.append({
            "차수": rnd,
            "ELO": current_elo
        })
    
    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        
        # 4. 그래프 생성
        fig = px.line(df_plot, x='차수', y='ELO', 
                     title=f"📈 {player_name} 도토리",
                     markers=True,
                     text=df_plot['ELO'].apply(lambda x: f"{int(x)}"))
        
        # Y축 범위 고정 (데이터 중 최솟값/최댓값을 고려하거나 특정 범위로 고정)
        # 예: 1300점에서 1700점 사이로 고정 (필요시 값 조정 가능)
        fig.update_yaxes(range=[1400, 1600]) 
        
        fig.update_traces(textposition="top center", line_shape='linear')
        fig.update_layout(
            xaxis_title="대회 차수", 
            yaxis_title="ELO 점수 (고정 축)",
            template='plotly_white',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"{player_name} 도토리의 데이터가 아직 없습니다.")

# --- 4. UI 구성 ---

st.set_page_config(page_title="도토리 키재기", page_icon="🎾", layout="wide")

if 'is_admin' not in st.session_state: st.session_state.is_admin = False

df_rank = get_ranking_statistics()
names = sorted(df_rank['이름'].tolist()) if not df_rank.empty else []

st.title("🎾 최고의 도토리는?")

with st.sidebar:
    st.header("🔐 관리자 접속")
    pwd = st.text_input("관리자 비밀번호", type="password")
    if pwd == ADMIN_PASSWORD:
        st.session_state.is_admin = True
        st.success("관리자 모드 활성화")
    else: st.session_state.is_admin = False

    if st.session_state.is_admin:
        st.divider()
        st.header("⚙️ 관리 도구")
        if st.button("🔄 전체 데이터 재정산"):
            with st.spinner("무승부 로직을 포함하여 재계산 중..."):
                recalculate_all_cloud_data()
                st.success("재계산 완료!")
                st.rerun()
        
        with st.expander("👤 도토리 등록"):
            new_p = st.text_input("이름")
            if st.button("즉시 등록"):
                if new_p and add_new_player(new_p):
                    st.success(f"{new_p} 등록됨"); st.rerun()
                else: st.error("이미 존재하거나 오류가 발생했습니다.")

tab_names = ["📊 대시보드", "🏆 상세 랭킹", "📜 경기 이력"]
if st.session_state.is_admin: tab_names.insert(1, "📝 결과 입력")
tabs = st.tabs(tab_names)

# 대시보드
with tabs[0]:
    df_rank = get_ranking_statistics()
    df_m_all = get_matches()

    if not df_rank.empty:
        # 1. 지표 데이터 준비
        # 경기수가 0보다 큰 선수만 카운트
        active_players = len(df_rank[df_rank['경기수'] > 0])
        
        # 총 차수 (중복 제외 차수 개수)
        total_rounds = df_m_all['round'].nunique() if not df_m_all.empty else 0
        
        # 최고 도토리 (1위 이름과 점수 결합)
        top_player = df_rank.iloc[0]
        # top_info = f"{top_player['이름']} ({top_player['ELO 점수']}pt)"

        # 2. UI 출력 (4열 구성)
        c1, c2, c3, c4 = st.columns(4)
        
        c1.metric("👤 참가 도토리", f"{active_players}명")
        c2.metric("📅 총 차수", f"{total_rounds}차")
        c3.metric("🎾 누적 경기", f"{len(df_m_all)}회")
        c4.metric("🏆 최고 도토리", f"{top_player['이름']}")

    st.divider()       
    
    st.subheader("📊 도토리 키재기")
    # df_rank가 비어있지 않을 때만 그래프를 그립니다.
    if not df_rank.empty:
        # 데이터가 10개보다 적을 수도 있으므로 안전하게 head(10)
        fig = px.bar(df_rank.head(10), x='이름', y='ELO 점수', color='ELO 점수', text='ELO 점수',
                    color_continuous_scale='Viridis', template='plotly_white')
        
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_title="도토리", yaxis_title="ELO 점수", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 데이터가 없을 때 표시할 문구
        st.info("💡 아직 등록된 경기 결과가 없어 그래프를 표시할 수 없습니다. 관리자 탭에서 도토리를 등록하고 경기를 입력해 주세요!")

    st.divider()

    
    st.subheader("🔍 도토리 실력 추이")

    # 랭킹에 있는 이름 리스트 사용
    if not df_rank.empty:
        player_to_show = st.selectbox("그래프로 확인할 도토리를 선택하세요", df_rank['이름'].tolist())
        if player_to_show:
            display_player_trend(player_to_show)
    else:
        # 데이터가 없을 때 표시할 문구
        st.info("💡 아직 등록된 경기 결과가 없어 그래프를 표시할 수 없습니다. 관리자 탭에서 도토리를 등록하고 경기를 입력해 주세요!")

# 결과 입력 (관리자 전용)
if st.session_state.is_admin:
# 탭: 결과 입력 (관리자용)
    with tabs[1]:
        # 차수 자동 계산
        default_next_round = get_next_round_name()
        
        # --- [A] 수동 경기 입력 섹션 ---
        st.subheader("✍️ 수동 경기 입력")
        with st.expander("한 경기씩 직접 입력하기", expanded=False):
            with st.form("manual_input_form"):
                m_round = st.text_input("차수 정보", value=default_next_round)
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    m_w1 = st.selectbox("승자 1", names, key="m_w1")
                    m_w2 = st.selectbox("승자 2", names, key="m_w2")
                with col2:
                    m_l1 = st.selectbox("패자 1", names, key="m_l1")
                    m_l2 = st.selectbox("패자 2", names, key="m_l2")
                with col3:
                    m_score = st.text_input("점수", value="6:0")
                
                # 수동 입력 폼 내부 (기존 코드 대체)
                if st.form_submit_button("경기 저장"):
                    if len(set([m_w1, m_w2, m_l1, m_l2])) < 4:
                        st.error("도토리가 중복되었습니다.")
                    else:
                        # ⭐ 원칙 적용: 현재 점수가 아닌 '차수 시작 점수' 가져오기
                        start_elos = get_round_start_elos(m_round)
                        
                        w_avg = (start_elos.get(m_w1, 1500) + start_elos.get(m_w2, 1500)) / 2
                        l_avg = (start_elos.get(m_l1, 1500) + start_elos.get(m_l2, 1500)) / 2
                        
                        exp, diff = calculate_elo_logic(w_avg, l_avg, m_score)
                        
                        save_match_to_cloud({
                            "round": m_round,
                            "winner1": m_w1, "winner2": m_w2,
                            "loser1": m_l1, "loser2": m_l2,
                            "score": m_score,
                            "elo_change": diff,
                            "expected_win": exp,
                            "timestamp": datetime.now().isoformat()
                        },w_avg,l_avg)
                        st.success(f"✅ {m_round} 경기 저장 완료!")
                        st.rerun()
        st.divider()
 
 
    # with tabs[1]:
        st.subheader("📸 AI 경기 결과 자동 입력")
        uploaded_file = st.file_uploader("경기 결과 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            img_bytes = uploaded_file.read()
            st.image(img_bytes, caption="업로드된 이미지", use_container_width=True)
            
            if st.button("🪄 AI 분석 시작"):
                with st.spinner("AI가 경기 결과를 판독하고 있습니다..."):
                    res = analyze_image_with_ai(img_bytes)
                    if res:
                        st.session_state.ai_res = res
                        st.success("분석 완료! 아래 결과를 확인하고 수정하세요.")
                    else:
                        st.error("분석에 실패했습니다. 수동 입력을 이용하거나 이미지를 다시 확인하세요.")

        # AI 분석 결과가 세션에 있을 때 표시
        if st.session_state.get('ai_res'):
            st.divider()
            st.subheader("📝 분석 결과 검토 및 저장")
            
            with st.form("ai_save_form"):
                target_round = st.text_input("차수 정보", value=default_next_round)
                match_list = st.session_state.ai_res.get("match_list", [])
                
                updated_matches = []
                # 현재 등록된 전체 선수 명단 가져오기
                current_players_df = get_players()
                registered_names = current_players_df['name'].tolist()
                
                for i, m in enumerate(match_list):
                    st.markdown(f"**[경기 {i+1}]**")
                    c1, c2, c3 = st.columns([2, 2, 1])
                    
                    # AI가 인식한 이름들
                    raw_w1, raw_w2 = m.get('winner1', ''), m.get('winner2', '')
                    raw_l1, raw_l2 = m.get('loser1', ''), m.get('loser2', '')
                    
                    # UI용 리스트 생성 (기존 명단 + AI가 새로 찾은 이름 합치기)
                    temp_names = sorted(list(set(registered_names + [raw_w1, raw_w2, raw_l1, raw_l2])))
                    
                    with c1:
                        mw1 = st.selectbox(f"승자1", temp_names, index=temp_names.index(raw_w1) if raw_w1 in temp_names else 0, key=f"w1_{i}")
                        mw2 = st.selectbox(f"승자2", temp_names, index=temp_names.index(raw_w2) if raw_w2 in temp_names else 0, key=f"w2_{i}")
                    with c2:
                        ml1 = st.selectbox(f"패자1", temp_names, index=temp_names.index(raw_l1) if raw_l1 in temp_names else 0, key=f"l1_{i}")
                        ml2 = st.selectbox(f"패자2", temp_names, index=temp_names.index(raw_l2) if raw_l2 in temp_names else 0, key=f"l2_{i}")
                    with c3:
                        msc = st.text_input(f"점수", m.get('score', '6:0'), key=f"sc_{i}")
                    
                    updated_matches.append({"w1": mw1, "w2": mw2, "l1": ml1, "l2": ml2, "score": msc})

                submit = st.form_submit_button("🚀 경기 결과 저장")
                
                if submit:
                    # 1. 신규 회원 자동 등록 프로세스
                    all_input_names = []
                    for um in updated_matches:
                        all_input_names.extend([um['w1'], um['w2'], um['l1'], um['l2']])
                    
                    new_count = 0
                    for name in set(all_input_names):
                        if name and name not in registered_names:
                            if add_new_player(name):
                                new_count += 1
                    
                    if new_count > 0:
                        st.info(f"🆕 {new_count}명의 신규 회원이 자동으로 등록되었습니다.")

                    # 2. ⭐ 원칙 적용: 차수 시작 시점의 ELO 스냅샷 확보
                    start_elos = get_round_start_elos(target_round)

                    success_count = 0
                    for um in updated_matches:
                        # 스냅샷 점수를 사용하여 모든 경기를 동일한 기준으로 계산
                        w_avg = (start_elos.get(um['w1'], 1500) + start_elos.get(um['w2'], 1500)) / 2
                        l_avg = (start_elos.get(um['l1'], 1500) + start_elos.get(um['l2'], 1500)) / 2
                        
                        exp, diff = calculate_elo_logic(w_avg, l_avg, um['score'])
                        
                        save_match_to_cloud({
                            "round": target_round,
                            "winner1": um['w1'], "winner2": um['w2'],
                            "loser1": um['l1'], "loser2": um['l2'],
                            "score": um['score'],
                            "elo_change": diff,
                            "expected_win": exp,
                            "timestamp": datetime.now().isoformat()
                        }, w_avg, l_avg)
                        success_count += 1
                    
                    st.success(f"✅ {success_count}개의 경기가 저장되었습니다!")
                    st.session_state.ai_res = None # 처리 완료 후 초기화
                    st.rerun()

# 탭: 상세 랭킹
idx_rank = 2 if st.session_state.is_admin else 1
# 상세 랭킹 탭 (tabs[idx_rank])
with tabs[idx_rank]:
    st.subheader("🏆 도토리 랭킹")
    df_rank = get_ranking_statistics()
    
    if not df_rank.empty:
        display_df = df_rank.copy()
        # 승률과 출석률에 % 기호 붙이기
        display_df['승률'] = display_df['승률'].astype(str) + "%"
        display_df['출석률'] = display_df['출석률'].astype(str) + "%" # 신규 추가
        
        styled_rank = display_df.style.set_properties(**{
            'text-align': 'center',
            'vertical-align': 'middle'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('text-align', 'center'), 
                ('background-color', '#f0f2f6'), 
                ('color', '#31333f'),
                ('font-weight', 'bold')
            ]}
        ]).hide(axis='index')
        
        st.write(styled_rank.to_html(), unsafe_allow_html=True) #
        
    else:
        st.info("도토리 데이터가 없습니다.")

# 탭: 경기 이력 (tabs[idx_hist])
idx_hist = 3 if st.session_state.is_admin else 2
with tabs[idx_hist]:
    st.subheader("📜 매치 히스토리")
    df_history = get_matches()
    
    if not df_history.empty:
        # 1. 정렬: 차수 내림차순 -> 시간 내림차순
        df_history['round_val'] = df_history['round'].apply(extract_round_number)
        df_history = df_history.sort_values(by=['round_val', 'timestamp'], ascending=[False, False])
        
        # 2. 데이터 가공 (평균 ELO 및 기대승률)
        df_p = get_players()
        p_elo_dict = dict(zip(df_p['name'], df_p['elo']))
        
        def process_row(row):
            # 1. DB에 '당시 계산 기준 점수'가 저장되어 있다면 최우선으로 사용
            w_avg = row.get('w_avg_at_match')
            l_avg = row.get('l_avg_at_match')
            
            # 2. 저장된 값이 없는 과거 데이터의 경우 (재계산 로직 보완)
            if pd.isna(w_avg) or w_avg is None:
                # 해당 차수의 시작 시점 점수를 가져오도록 함수 호출
                start_elos = get_round_start_elos(row['round'])
                
                w1_elo = start_elos.get(row['winner1'], 1500.0)
                w2_elo = start_elos.get(row['winner2'], 1500.0)
                w_avg = (w1_elo + w2_elo) / 2
                
                l1_elo = start_elos.get(row['loser1'], 1500.0)
                l2_elo = start_elos.get(row['loser2'], 1500.0)
                l_avg = (l1_elo + l2_elo) / 2
                
            win_exp_val = row.get('expected_win', 0.5)
            win_exp = f"{int(win_exp_val * 100)}%"
            
            return int(round(w_avg)), int(round(l_avg)), win_exp

        df_history[['승자평균', '패자평균', '승률']] = df_history.apply(
            lambda x: pd.Series(process_row(x)), axis=1
        )

         # 데이터 정리 시 '변동' 값을 소수점 한자리까지 포함된 실수형으로 유지
        display_cols = ['round', 'winner1', 'winner2', '승자평균', 'score', 'loser1', 'loser2', '패자평균', 'elo_change', '승률']
        rename_map = {
            'round': '차수', 'winner1': '승자1', 'winner2': '승자2', 'score': '결과',
            'loser1': '패자1', 'loser2': '패자2', 'elo_change': '변동', '승률': '승리확률'
        }
        final_df = df_history[display_cols].rename(columns=rename_map)

        # 1. 차수별 배경색 함수 (기존 동일)
        def style_by_round(row):
            rnd_val = extract_round_number(row['차수'])
            return ['background-color: #ffffff'] * len(row) if rnd_val % 2 == 0 else ['background-color: #f9fbfd'] * len(row)

        # 2. 변동폭 색상 강조 및 소수점 처리 함수
        def color_variant(val):
            if isinstance(val, (int, float)):
                color = '#e74c3c' if val > 0 else '#3498db' if val < 0 else 'black'
                return f'color: {color}; font-weight: bold;'
            return 'color: black'

        # 3. Pandas Styler 적용 (format 추가)
        styled_hist = final_df.style.apply(style_by_round, axis=1) \
            .applymap(color_variant, subset=['변동']) \
            .format({'변동': "{:+.1f}"}) \
            .set_properties(**{
                'text-align': 'center',
                'vertical-align': 'middle',
                'padding': '12px 4px',
                'border-bottom': '1px solid #f0f0f0'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('text-align', 'center'), 
                    ('background-color', '#edf2f7'), 
                    ('color', '#2d3748'),
                    ('font-weight', 'bold'),
                    ('border-bottom', '2px solid #cbd5e0')]
                },
                {'selector': '', 'props': [
                    ('width', '100%'), 
                    ('border-collapse', 'collapse'),
                    ('border', '1px solid #e2e8f0')]
                }
            ]).hide(axis='index')

        # 4. HTML 출력
        st.write(styled_hist.to_html(escape=False), unsafe_allow_html=True)

    else:
        st.info("기록된 경기 이력이 없습니다.")