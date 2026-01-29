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
APP_ID = "tennis_club_v1"

ADMIN_PASSWORD = "ace_admin!" 
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025" 
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. Firestore 데이터 조작 함수 ---

def get_players():
    """Firestore에서 선수 명단을 실시간으로 가져옵니다."""
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

def save_match_to_cloud(match_data):
    """경기 결과 저장 및 ELO 점수 원자적 업데이트"""
    if not db: return
    
    matches_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'matches')
    matches_ref.add(match_data)
    
    change = match_data['elo_change']
    winners = [match_data['winner1'], match_data['winner2']]
    losers = [match_data['loser1'], match_data['loser2']]
    
    # 승자 그룹(Winner1, 2)에 변동폭 반영
    for w in winners:
        p_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(w)
        p_ref.update({"elo": firestore.Increment(change)})
    # 패자 그룹(Loser1, 2)에 반대 변동폭 반영
    for l in losers:
        p_ref = db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(l)
        p_ref.update({"elo": firestore.Increment(-change)})

def add_new_player(name):
    """새 선수 등록"""
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
    """전체 데이터 재정산 로직"""
    if not db: return
    df_p = get_players()
    df_m = get_matches()
    if df_m.empty: return

    df_m['round_val'] = df_m['round'].apply(extract_round_number)
    df_m = df_m.sort_values(by=['round_val', 'timestamp']).reset_index(drop=True)
    
    working_elos = {name: 1500.0 for name in df_p['name']}
    
    for idx, row in df_m.iterrows():
        w1, w2, l1, l2 = row['winner1'], row['winner2'], row['loser1'], row['loser2']
        w_avg = (working_elos.get(w1, 1500.0) + working_elos.get(w2, 1500.0)) / 2
        l_avg = (working_elos.get(l1, 1500.0) + working_elos.get(l2, 1500.0)) / 2
        
        exp, diff = calculate_elo_logic(w_avg, l_avg, row['score'])
        
        match_doc = db.collection('artifacts', APP_ID, 'public', 'data', 'matches').document(row['id'])
        match_doc.update({"elo_change": diff, "expected_win": exp})
        
        for w in [w1, w2]: 
            if w in working_elos: working_elos[w] += diff
        for l in [l1, l2]: 
            if l in working_elos: working_elos[l] -= diff

    for name, final_elo in working_elos.items():
        db.collection('artifacts', APP_ID, 'public', 'data', 'players').document(name).update({"elo": final_elo})

def get_ranking_statistics():
    """상세 랭킹 데이터 산출"""
    df_p = get_players()
    df_m = get_matches()
    if df_p.empty: return pd.DataFrame()
    
    stats = {n: {'이름': n, 'ELO 점수': e, '승': 0, '무': 0, '패': 0, '득': 0, '실': 0} 
             for n, e in zip(df_p['name'], df_p['elo'])}
    
    if not df_m.empty:
        for _, row in df_m.iterrows():
            w1, w2, l1, l2 = row['winner1'], row['winner2'], row['loser1'], row['loser2']
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
        total = s['승'] + s['무'] + s['패']
        wr = (s['승'] / total * 100) if total > 0 else 0
        res.append({
            '이름': n, 'ELO 점수': int(round(s['ELO 점수'])),
            '승': s['승'], '무': s['무'], '패': s['패'], 
            '득실': s['득'] - s['실'], '승률': int(round(wr))
        })
    
    df = pd.DataFrame(res)
    if not df.empty:
        df = df.sort_values(['ELO 점수', '승', '무', '득실', '이름'], 
                            ascending=[False, False, False, False, True]).reset_index(drop=True)
        df.insert(0, '순위', range(1, len(df) + 1))
    return df

def analyze_image_with_ai(image_bytes):
    """Gemini AI 분석"""
    if not API_KEY: return {"error": "Gemini API 키가 설정되지 않았습니다."}
    try:
        client = genai.Client(api_key=API_KEY)
        prompt = "Extract tennis match info in JSON: player_list (names), match_list (winner1, winner2, loser1, loser2, score)."
        response = client.models.generate_content(
            model=GEMINI_MODEL, 
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_text(text=prompt), 
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                ])
            ]
        )
        json_str = response.text.strip()
        if "```" in json_str: 
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"): json_str = json_str[4:].strip()
        return json.loads(json_str)
    except Exception as e: 
        return {"error": f"AI 분석 오류: {str(e)}"}

# --- 4. UI 구성 ---

st.set_page_config(page_title="테니스 매니저 AI Pro (Cloud)", page_icon="🎾", layout="wide")

if 'is_admin' not in st.session_state: st.session_state.is_admin = False

df_rank = get_ranking_statistics()
names = sorted(df_rank['이름'].tolist()) if not df_rank.empty else []

st.title("🎾 평촌에이스 최고수는 누굴까?")

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
        
        with st.expander("👤 신규 선수 등록"):
            new_p = st.text_input("선수명")
            if st.button("즉시 등록"):
                if new_p and add_new_player(new_p):
                    st.success(f"{new_p} 등록됨"); st.rerun()
                else: st.error("이미 존재하거나 오류가 발생했습니다.")

tab_names = ["📊 대시보드", "🏆 상세 랭킹", "📜 경기 이력"]
if st.session_state.is_admin: tab_names.insert(1, "📝 결과 입력")
tabs = st.tabs(tab_names)

# 대시보드
with tabs[0]:
    if not df_rank.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 등록 인원", f"{len(df_rank)}명")
        c2.metric("현재 1위", df_rank.iloc[0]['이름'])
        c3.metric("평균 ELO", f"{int(df_rank['ELO 점수'].mean())} pts")
        df_m_all = get_matches()
        c4.metric("총 경기 수", f"{len(df_m_all)}회")
        
        st.subheader("📊 상위 10인 실력 분포")
        fig = px.bar(df_rank.head(10), x='이름', y='ELO 점수', color='ELO 점수', text='ELO 점수', 
                     color_continuous_scale='Viridis', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("데이터가 없습니다.")

# 결과 입력 (관리자 전용)
if st.session_state.is_admin:
    with tabs[1]:
        mode = st.radio("입력 방식", ["수동 직접 입력", "AI 이미지 분석"], horizontal=True)
        
        if mode == "AI 이미지 분석":
            up_file = st.file_uploader("기록지 이미지 업로드", type=['jpg', 'jpeg', 'png'])
            if up_file and st.button("AI 분석 시작"):
                with st.spinner("AI 분석 중..."):
                    res = analyze_image_with_ai(up_file.getvalue())
                    if "error" not in res:
                        st.session_state.ai_res = res
                        st.success("분석 완료!")
                    else: st.error(res['error'])
            
            if st.session_state.get('ai_res'):
                with st.form("ai_save_form"):
                    target_rnd = st.text_input("차수 정보", "1차 정기전")
                    match_list = st.session_state.ai_res.get("match_list", [])
                    for i, m in enumerate(match_list):
                        st.markdown(f"--- Match {i+1}")
                        col1, col2, col3 = st.columns([2, 2, 1])
                        idx_w1 = names.index(m['winner1']) if m.get('winner1') in names else 0
                        idx_w2 = names.index(m['winner2']) if m.get('winner2') in names else 0
                        idx_l1 = names.index(m['loser1']) if m.get('loser1') in names else 0
                        idx_l2 = names.index(m['loser2']) if m.get('loser2') in names else 0
                        w1 = col1.selectbox("승자(A팀)1", names, index=idx_w1, key=f"ai_w1_{i}")
                        w2 = col1.selectbox("승자(A팀)2", names, index=idx_w2, key=f"ai_w2_{i}")
                        l1 = col2.selectbox("패자(B팀)1", names, index=idx_l1, key=f"ai_l1_{i}")
                        l2 = col2.selectbox("패자(B팀)2", names, index=idx_l2, key=f"ai_l2_{i}")
                        sc = col3.text_input("점수", m.get('score', '6:0'), key=f"ai_sc_{i}")
                    
                    if st.form_submit_button("모든 분석 결과 저장"):
                        df_p_now = get_players()
                        elo_map = dict(zip(df_p_now['name'], df_p_now['elo']))
                        for i in range(len(match_list)):
                            sw1, sw2 = st.session_state[f"ai_w1_{i}"], st.session_state[f"ai_w2_{i}"]
                            sl1, sl2 = st.session_state[f"ai_l1_{i}"], st.session_state[f"ai_l2_{i}"]
                            ssc = st.session_state[f"ai_sc_{i}"]
                            w_avg = (elo_map.get(sw1, 1500) + elo_map.get(sw2, 1500)) / 2
                            l_avg = (elo_map.get(sl1, 1500) + elo_map.get(sl2, 1500)) / 2
                            exp, diff = calculate_elo_logic(w_avg, l_avg, ssc)
                            save_match_to_cloud({"round": target_rnd, "winner1": sw1, "winner2": sw2, "loser1": sl1, "loser2": sl2, "score": ssc, "elo_change": diff, "expected_win": exp, "timestamp": datetime.now().isoformat()})
                        st.session_state.ai_res = None
                        st.success("Firestore 저장 완료!"); st.rerun()

        else:
            with st.form("manual_form"):
                round_n = st.text_input("차수/대회명", "정기전")
                c1, c2, c3 = st.columns([2, 2, 1])
                mw1 = c1.selectbox("승자(A팀)1", names); mw2 = c1.selectbox("승자(A팀)2", names)
                ml1 = c2.selectbox("패자(B팀)1", names); ml2 = c2.selectbox("패자(B팀)2", names)
                msc = c3.text_input("최종 점수", "6:6")
                
                if st.form_submit_button("경기 결과 저장"):
                    df_p_now = get_players()
                    elo_map = dict(zip(df_p_now['name'], df_p_now['elo']))
                    w_avg = (elo_map.get(mw1, 1500) + elo_map.get(mw2, 1500)) / 2
                    l_avg = (elo_map.get(ml1, 1500) + elo_map.get(ml2, 1500)) / 2
                    exp, diff = calculate_elo_logic(w_avg, l_avg, msc)
                    save_match_to_cloud({"round": round_n, "winner1": mw1, "winner2": mw2, "loser1": ml1, "loser2": ml2, "score": msc, "elo_change": diff, "expected_win": exp, "timestamp": datetime.now().isoformat()})
                    st.success("클라우드 저장 완료!"); st.rerun()

# 탭: 상세 랭킹
idx_rank = 2 if st.session_state.is_admin else 1
with tabs[idx_rank]:
    st.subheader("🏆 전체 선수 랭킹")
    if not df_rank.empty:
        display_df = df_rank.copy()
        display_df['승률'] = display_df['승률'].astype(str) + "%"
        styled_html = display_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f0f2f6'), ('color', '#31333f')]}
        ]).hide(axis='index').to_html()
        st.write(styled_html, unsafe_allow_html=True)

# 탭: 경기 이력
idx_hist = 3 if st.session_state.is_admin else 2
with tabs[idx_hist]:
    st.subheader("📜 매치 히스토리 (최신순)")
    df_history = get_matches()
    if not df_history.empty:
        df_history['sort_val'] = df_history['round'].apply(extract_round_number)
        df_history = df_history.sort_values(by=['sort_val', 'timestamp'], ascending=[False, False])
        for _, r in df_history.iterrows():
            with st.expander(f"[{r['round']}] {r['winner1']}·{r['winner2']} vs {r['loser1']}·{r['loser2']} ({r['score']})"):
                st.write(f"**ELO 변동:** {r['elo_change']:+.1f} pts | **기대 승률:** {r.get('expected_win', 0)*100:.1f}%")
                st.caption(f"기록일시: {r['timestamp']}")