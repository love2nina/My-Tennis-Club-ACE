import streamlit as st
import firebase_admin
import json
from firebase_admin import credentials, firestore
import datetime

# Firebase 초기화 (이미 되어 있다면 이 부분은 건너뛰세요)
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

# --- 도움 함수 ---
def is_registration_open():
    """settings 컬렉션에서 현재 신청 가능 여부를 가져옴"""
    doc = db.collection("settings").document("config").get()
    if doc.exists:
        return doc.to_dict().get("is_open", True)
    return True

def check_player_exists(name):
    """players 컬렉션에 해당 이름의 문서가 있는지 확인"""
    doc = db.collection("players").document(name).get()
    return doc.exists

# --- UI 구현 ---
st.title("🎾 테니스 동호회 참가 신청")

# 1. 현재 신청 상태 확인
open_status = is_registration_open()

if open_status:
    st.info("📢 현재 이번 주 경기 참가 신청을 받고 있습니다.")
    
    with st.form("match_apply_form", clear_on_submit=True):
        player_name = st.text_input("성함 (DB에 등록된 실명 입력)")
        user_memo = st.text_input("비고 (특이사항이 있다면 적어주세요)")
        
        submit_button = st.form_submit_button("참가 신청하기")
        
        if submit_button:
            if not player_name:
                st.error("이름을 입력해 주세요.")
            else:
                # 2. 회원 명단 존재 여부 확인
                if check_player_exists(player_name):
                    # 3. applicants 컬렉션에 데이터 저장
                    db.collection("applicants").document(player_name).set({
                        "name": player_name,
                        "memo": user_memo,
                        "applied_at": datetime.datetime.now()
                    })
                    st.success(f"✅ {player_name}님, 신청이 완료되었습니다!")
                else:
                    st.error("❌ 등록되지 않은 회원 이름입니다. 실명을 입력하시거나 운영진에게 문의하세요.")
else:
    st.warning("🚫 현재 참가 신청 기간이 아닙니다. 마감 후 대진표를 확인해 주세요.")


    # 위 코드 하단에 추가하거나 별도 탭으로 구성
st.divider()
with st.expander("🛠 운영자 전용 (신청 관리)"):
    admin_password = st.text_input("관리자 비밀번호", type="password")
    if admin_password == "your_password": # 실제 사용할 비밀번호 설정
        current_status = is_registration_open()
        
        if current_status:
            if st.button("🔴 참가 신청 마감하기"):
                db.collection("settings").document("config").update({"is_open": False})
                st.rerun()
        else:
            if st.button("🟢 참가 신청 다시 열기"):
                db.collection("settings").document("config").update({"is_open": True})
                st.rerun()