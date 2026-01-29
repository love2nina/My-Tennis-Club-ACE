import streamlit as st
import sqlite3
import pandas as pd
import os
import base64
from datetime import datetime
import json
import re
import plotly.express as px
from google import genai
from google.genai import types

# --- 데이터베이스 및 설정 ---
DB_FILE = 'tennis_club.db'

# 관리자 비밀번호 설정 (원하는 비밀번호로 변경하세요)
ADMIN_PASSWORD = "ace_admin!" 

# Gemini API 설정 (Streamlit Secrets에서 가져옴)
GEMINI_MODEL = "gemini-1.5-flash-latest" 
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

def init_db():
    """데이터베이스와 테이블을 초기화합니다."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS players 
                 (name TEXT PRIMARY KEY, elo REAL DEFAULT 1500.0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS matches 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, round TEXT, 
                  winner1 TEXT, winner2 TEXT, loser1 TEXT, loser2 TEXT, 
                  score TEXT, elo_change REAL, expected_win REAL,
                  image_data BLOB, match_detail_json TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def extract_round_number(round_str):
    """차수 문자열에서 숫자를 추출하여 정렬 기준값으로 반환합니다."""
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", round_str)
    return float(numbers[0]) if numbers else 0.0

def get_base_elos_for_round(target_round):
    """입력된 차수보다 '작은' 차수의 경기들만 합산하여 기준 ELO를 계산합니다."""
    conn = sqlite3.connect(DB_FILE)
    players = pd.read_sql("SELECT name FROM players", conn)
    matches = pd.read_sql("SELECT round, winner1, winner2, loser1, loser2, elo_change FROM matches", conn)
    conn.close()

    target_val = extract_round_number(target_round)
    elo_dict = {name: 1500.0 for name in players['name']}

    for _, row in matches.iterrows():
        match_val = extract_round_number(row['round'])
        if match_val < target_val:
            change = row['elo_change']
            if row['winner1'] in elo_dict: elo_dict[row['winner1']] += change
            if row['winner2'] in elo_dict: elo_dict[row['winner2']] += change
            if row['loser1'] in elo_dict: elo_dict[row['loser1']] -= change
            if row['loser2'] in elo_dict: elo_dict[row['loser2']] -= change
            
    return elo_dict

def get_ranking_statistics():
    """랭킹 산정을 위한 상세 통계를 계산합니다."""
    conn = sqlite3.connect(DB_FILE)
    players = pd.read_sql("SELECT name FROM players", conn)
    matches = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()

    stats = {}
    for name in players['name']:
        stats[name] = {'이름': name, 'ELO 점수': 1500.0, '승': 0, '무': 0, '패': 0, '득': 0, '실': 0}

    for _, row in matches.iterrows():
        w1, w2, l1, l2 = row['winner1'], row['winner2'], row['loser1'], row['loser2']
        change = row['elo_change']
        for w in [w1, w2]:
            if w in stats:
                stats[w]['ELO 점수'] += change
                stats[w]['승'] += 1
        for l in [l1, l2]:
            if l in stats:
                stats[l]['ELO 점수'] -= change
                stats[l]['패'] += 1
        try:
            s_parts = row['score'].split(':')
            s_win, s_loss = int(s_parts[0]), int(s_parts[1])
            if s_win == s_loss:
                for p in [w1, w2, l1, l2]:
                    if p in stats: stats[p]['승'] -= 1; stats[p]['무'] += 1
            for w in [w1, w2]:
                if w in stats: stats[w]['득'] += s_win; stats[w]['실'] += s_loss
            for l in [l1, l2]:
                if l in stats: stats[l]['득'] += s_loss; stats[l]['실'] += s_win
        except: pass

    ranking_list = []
    for name, s in stats.items():
        total_games = s['승'] + s['무'] + s['패']
        win_rate_val = (s['승'] / total_games * 100) if total_games > 0 else 0
        ranking_list.append({
            '이름': name,
            'ELO 점수': int(round(s['ELO 점수'])), 
            '승': s['승'], '무': s['무'], '패': s['패'],
            '득실': s['득'] - s['실'], '승률': int(round(win_rate_val))
        })

    df = pd.DataFrame(ranking_list)
    if not df.empty:
        df = df.sort_values(by=['ELO 점수', '승', '무', '득실', '승률', '이름'], ascending=[False, False, False, False, False, True]).reset_index(drop=True)
        df.insert(0, '순위', range(1, len(df) + 1))
        return df
    return pd.DataFrame()

def calculate_elo_logic(w_avg, l_avg, score_text, k=32):
    expect_win = 1 / (10 ** ((l_avg - w_avg) / 400) + 1)
    try:
        if ':' in score_text:
            s1, s2 = map(int, score_text.split(':'))
            score_diff = abs(s1 - s2)
        else: score_diff = 1
    except: score_diff = 1 
    
    if score_diff >= 6: multiplier = 1.5
    elif score_diff >= 4: multiplier = 1.25
    elif score_diff >= 2: multiplier = 1.0
    else: multiplier = 0.8
    
    change = k * (1 - expect_win) * multiplier
    return round(expect_win, 4), round(change, 2)

def recalculate_all_matches():
    conn = sqlite3.connect(DB_FILE)
    players_data = conn.execute("SELECT name FROM players").fetchall()
    players = [r[0] for r in players_data]
    matches = pd.read_sql("SELECT * FROM matches", conn)
    if matches.empty:
        conn.close()
        return

    matches['round_val'] = matches['round'].apply(extract_round_number)
    matches = matches.sort_values(by=['round_val', 'id']).reset_index(drop=True)
    
    unique_rounds = sorted(matches['round_val'].unique())
    temp_working_elos = {name: 1500.0 for name in players}
    round_base_elos = {}
    
    for r_val in unique_rounds:
        round_base_elos[r_val] = temp_working_elos.copy()
        round_matches = matches[matches['round_val'] == r_val]
        for _, rm in round_matches.iterrows():
            change = rm['elo_change']
            if rm['winner1'] in temp_working_elos: temp_working_elos[rm['winner1']] += change
            if rm['winner2'] in temp_working_elos: temp_working_elos[rm['winner2']] += change
            if rm['loser1'] in temp_working_elos: temp_working_elos[rm['loser1']] -= change
            if rm['loser2'] in temp_working_elos: temp_working_elos[rm['loser2']] -= change

    for idx, row in matches.iterrows():
        r_val = row['round_val']
        base = round_base_elos[r_val]
        w1, w2, l1, l2 = row['winner1'], row['winner2'], row['loser1'], row['loser2']
        w_avg = (base.get(w1, 1500.0) + base.get(w2, 1500.0)) / 2
        l_avg = (base.get(l1, 1500.0) + base.get(l2, 1500.0)) / 2
        exp, diff = calculate_elo_logic(w_avg, l_avg, row['score'])
        conn.execute("UPDATE matches SET elo_change = ?, expected_win = ? WHERE id = ?", (diff, exp, int(row['id'])))
    conn.commit()
    conn.close()

def analyze_image_with_ai(image_bytes):
    if not API_KEY: return {"error": "API 키가 설정되지 않았습니다."}
    try:
        client = genai.Client(api_key=API_KEY)
        prompt = "테니스 경기 결과 기록지 이미지입니다. 다음 정보를 반드시 JSON 형식으로만 추출하세요: player_list, match_list (winner1, winner2, loser1, loser2, score)"
        response = client.models.generate_content(model=GEMINI_MODEL, contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt), types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")])])
        if response.text:
            json_str = response.text.strip()
            if "```" in json_str: 
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"): json_str = json_str[4:].strip()
            return json.loads(json_str)
        return {"error": "AI 응답 없음"}
    except Exception as e: return {"error": str(e)}

# --- UI 설정 ---
st.set_page_config(page_title="테니스 매니저 Pro", page_icon="🎾", layout="wide")
init_db()

# 세션 상태 초기화 (로그인 여부)
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# 데이터 로드
df_rank = get_ranking_statistics()
names = sorted(df_rank['이름'].tolist()) if not df_rank.empty else []

st.title("🎾 테니스 매니저 AI Pro")

with st.sidebar:
    st.header("🔐 관리자 접속")
    pwd_input = st.text_input("관리자 비밀번호", type="password")
    if pwd_input == ADMIN_PASSWORD:
        st.session_state.is_admin = True
        st.success("관리자 모드 활성화됨")
    else:
        st.session_state.is_admin = False
        if pwd_input:
            st.error("비밀번호가 틀렸습니다.")

    if st.session_state.is_admin:
        st.divider()
        st.header("⚙️ 관리 메뉴")
        if st.button("🔄 데이터 전체 재정산"):
            with st.spinner("재정산 중..."):
                recalculate_all_matches()
                st.success("재정산 완료!")
                st.rerun()
        with st.expander("👤 신규 선수 등록"):
            new_name = st.text_input("이름")
            if st.button("등록"):
                if new_name:
                    conn = sqlite3.connect(DB_FILE)
                    try:
                        conn.execute("INSERT INTO players (name) VALUES (?)", (new_name,))
                        conn.commit()
                        st.rerun()
                    except: st.error("이미 존재합니다.")
                    finally: conn.close()

# 탭 구성 (관리자 여부에 따라 탭 개수가 달라짐)
tab_names = ["📊 대시보드", "🏆 상세 랭킹", "📜 경기 이력"]
if st.session_state.is_admin:
    tab_names.insert(1, "📝 결과 입력") # 관리자일 때만 두 번째 위치에 삽입

tabs = st.tabs(tab_names)

# 탭 인덱스 매핑 (관리자 여부에 따라 변동 가능)
idx_dash = 0
idx_input = 1 if st.session_state.is_admin else -1
idx_rank = 2 if st.session_state.is_admin else 1
idx_history = 3 if st.session_state.is_admin else 2

# 탭: 대시보드
with tabs[idx_dash]:
    conn = sqlite3.connect(DB_FILE)
    matches_df = pd.read_sql("SELECT round FROM matches", conn)
    conn.close()
    total_rounds = matches_df['round'].nunique() if not matches_df.empty else 0
    total_matches = len(matches_df)
    
    if not df_rank.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 인원", f"{len(df_rank)}명")
        c2.metric("총 차수", f"{total_rounds}차")
        c3.metric("총 매치", f"{total_matches}회")
        top_player = df_rank.iloc[0]['이름']
        top_elo = df_rank.iloc[0]['ELO 점수']
        c4.metric("현재 랭킹 1위", top_player, f"{int(top_elo)} pts")
        
        st.subheader("📊 상위 10인 ELO 현황")
        top_10 = df_rank.head(10)
        fig = px.bar(top_10, x='이름', y='ELO 점수', color='ELO 점수', text='ELO 점수', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("선수를 먼저 등록해주세요.")

# 탭: 결과 입력 (관리자 전용)
if st.session_state.is_admin:
    with tabs[idx_input]:
        input_mode = st.radio("입력 방식 선택", ["AI 자동 분석", "수동 직접 입력"], horizontal=True)
        if input_mode == "AI 자동 분석":
            uploaded_file = st.file_uploader("이미지 업로드", type=['jpg', 'jpeg', 'png'])
            if uploaded_file and st.button("AI 분석 시작"):
                with st.spinner("분석 중..."):
                    ai_data = analyze_image_with_ai(uploaded_file.getvalue())
                    if "error" not in ai_data:
                        st.session_state.ai_result = ai_data
                        conn = sqlite3.connect(DB_FILE)
                        existing = [r[0] for r in conn.execute("SELECT name FROM players").fetchall()]
                        for p in ai_data.get("player_list", []):
                            p_name = p.get('name') if isinstance(p, dict) else p
                            if p_name and p_name not in existing:
                                conn.execute("INSERT INTO players (name) VALUES (?)", (p_name,))
                        conn.commit()
                        conn.close()
                        st.rerun()
                    else: st.error(ai_data['error'])

            if st.session_state.get('ai_result'):
                with st.form("save_match_ai"):
                    target_round = st.text_input("차수 입력", "1차")
                    current_elos_for_calc = get_base_elos_for_round(target_round)
                    save_list = []
                    for i, m in enumerate(st.session_state.ai_result.get("match_list", [])):
                        st.write(f"**Game {i+1}**")
                        col1, col2, col3 = st.columns([2, 2, 1])
                        w1 = col1.selectbox("승자1", names, index=names.index(m['winner1']) if m.get('winner1') in names else 0, key=f"ai_w1_{i}")
                        w2 = col1.selectbox("승자2", names, index=names.index(m['winner2']) if m.get('winner2') in names else 0, key=f"ai_w2_{i}")
                        l1 = col2.selectbox("패자1", names, index=names.index(m['loser1']) if m.get('loser1') in names else 0, key=f"ai_l1_{i}")
                        l2 = col2.selectbox("패자2", names, index=names.index(m['loser2']) if m.get('loser2') in names else 0, key=f"ai_l2_{i}")
                        sc = col3.text_input("점수", m.get("score", "6:0"), key=f"ai_sc_{i}")
                        w_avg = (current_elos_for_calc.get(w1, 1500) + current_elos_for_calc.get(w2, 1500)) / 2
                        l_avg = (current_elos_for_calc.get(l1, 1500) + current_elos_for_calc.get(l2, 1500)) / 2
                        exp, diff = calculate_elo_logic(w_avg, l_avg, sc)
                        save_list.append((w1, w2, l1, l2, sc, diff, exp))
                    if st.form_submit_button("모든 경기 기록 저장"):
                        conn = sqlite3.connect(DB_FILE)
                        for w1, w2, l1, l2, sc, df, ex in save_list:
                            conn.execute("INSERT INTO matches (round, winner1, winner2, loser1, loser2, score, elo_change, expected_win) VALUES (?,?,?,?,?,?,?,?)", (target_round, w1, w2, l1, l2, sc, df, ex))
                        conn.commit()
                        conn.close()
                        st.session_state.ai_result = None
                        st.success("데이터 저장 완료!")
                        st.rerun()
        else:
            with st.form("manual_input"):
                target_round = st.text_input("차수 입력", "1차")
                current_elos_for_calc = get_base_elos_for_round(target_round)
                c1, col2, col3 = st.columns([2, 2, 1])
                m_w1, m_w2 = c1.selectbox("승자1", names), c1.selectbox("승자2", names)
                m_l1, m_l2 = col2.selectbox("패자1", names), col2.selectbox("패자2", names)
                m_sc = col3.text_input("점수", "6:0")
                if st.form_submit_button("경기 결과 저장"):
                    w_avg = (current_elos_for_calc.get(m_w1, 1500) + current_elos_for_calc.get(m_w2, 1500)) / 2
                    l_avg = (current_elos_for_calc.get(m_l1, 1500) + current_elos_for_calc.get(m_l2, 1500)) / 2
                    exp, diff = calculate_elo_logic(w_avg, l_avg, m_sc)
                    conn = sqlite3.connect(DB_FILE)
                    conn.execute("INSERT INTO matches (round, winner1, winner2, loser1, loser2, score, elo_change, expected_win) VALUES (?,?,?,?,?,?,?,?)", (target_round, m_w1, m_w2, m_l1, m_l2, m_sc, diff, exp))
                    conn.commit()
                    conn.close()
                    st.success("저장 완료!")
                    st.rerun()

# 탭: 상세 랭킹
with tabs[idx_rank]:
    st.subheader("🏆 전체 선수 랭킹")
    if not df_rank.empty:
        display_df = df_rank.copy()
        display_df['승률'] = display_df['승률'].astype(str) + "%"
        styled_df = display_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f0f2f6')]} ]).hide(axis='index')
        st.write(styled_df.to_html(), unsafe_allow_html=True)
    else: st.info("기록된 경기 결과가 없습니다.")

# 탭: 경기 이력
with tabs[idx_history]:
    st.subheader("📜 매치 히스토리")
    conn = sqlite3.connect(DB_FILE)
    history = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()
    if not history.empty:
        history['round_sort_val'] = history['round'].apply(extract_round_number)
        history = history.sort_values(by=['round_sort_val', 'id'], ascending=[False, False])
        for _, r in history.iterrows():
            with st.expander(f"[{r['round']}] {r['winner1']}·{r['winner2']} vs {r['loser1']}·{r['loser2']} ({r['score']})"):
                st.write(f"변동폭: {r['elo_change']:+.1f} | 기대승률: {r['expected_win']*100:.1f}%")
                st.caption(f"기록일시: {r['timestamp']}")
    else: st.info("경기 이력이 없습니다.")