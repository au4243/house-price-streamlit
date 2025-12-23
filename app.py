import streamlit as st
import os
import json

from predict import HousePricePredictor

# =========================
# 頁面設定
# =========================
st.set_page_config(
    page_title="房價估價系統",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 房價估價與 SHAP 解釋系統")
st.caption("XGBoost + 可解釋 AI（SHAP）｜依據 114 年 Q1~Q3 不動產成交資料")

# =========================
# 行政區對照表
# =========================
CITY_TOWN_MAP = {
    "臺北市": [
        "士林區",
        "大同區",
        "大安區",
        "中山區",
        "中正區",
        "內湖區",
        "文山區",
        "北投區",
        "松山區",
        "信義區",
        "南港區",
        "萬華區",
    ],
    "新北市": [
        "八里區",
        "三芝區",
        "三重區",
        "三峽區",
        "土城區",
        "中和區",
        "五股區",
        "平溪區",
        "永和區",
        "石門區",
        "石碇區",
        "汐止區",
        "坪林區",
        "板橋區",
        "林口區",
        "金山區",
        "泰山區",
        "烏來區",
        "貢寮區",
        "淡水區",
        "深坑區",
        "新店區",
        "新莊區",
        "瑞芳區",
        "萬里區",
        "樹林區",
        "雙溪區",
        "蘆洲區",
        "鶯歌區",
    ],
    "桃園市": [
        "八德區",
        "大園區",
        "大溪區",
        "中壢區",
        "平鎮區",
        "桃園區",
        "復興區",
        "新屋區",
        "楊梅區",
        "龍潭區",
        "龜山區",
        "蘆竹區",
        "觀音區",
    ],
    "新竹縣": [
        "五峰鄉",
        "北埔鄉",
        "尖石鄉",
        "竹北市",
        "竹東鎮",
        "芎林鄉",
        "峨眉鄉",
        "湖口鄉",
        "新埔鎮",
        "新豐鄉",
        "橫山鄉",
        "關西鎮",
        "寶山鄉",
    ],    
    "苗栗縣": [
        "三義鄉",
        "三灣鄉",
        "大湖鄉",
        "公館鄉",
        "竹南鎮",
        "西湖鄉",
        "卓蘭鎮",
        "南庄鄉",
        "後龍鎮",
        "苑裡鎮",
        "苗栗市",
        "泰安鄉",
        "通霄鎮",
        "造橋鄉",
        "獅潭鄉",
        "銅鑼鄉",
        "頭份市",
        "頭屋鄉",
    ],    
    "臺中市": [
        "大甲區",
        "大安區",
        "大肚區",
        "大里區",
        "大雅區",
        "中區",
        "太平區",
        "北屯區",
        "北區",
        "外埔區",
        "石岡區",
        "后里區",
        "西屯區",
        "西區",
        "沙鹿區",
        "和平區",
        "東區",
        "東勢區",
        "南屯區",
        "南區",
        "烏日區",
        "神岡區",
        "梧棲區",
        "清水區",
        "新社區",
        "潭子區",
        "龍井區",
        "豐原區",
        "霧峰區",
    ],            
    "南投縣": [
        "中寮鄉",
        "仁愛鄉",
        "水里鄉",
        "名間鄉",
        "竹山鎮",
        "信義鄉",
        "南投市",
        "埔里鎮",
        "草屯鎮",
        "國姓鄉",
        "魚池鄉",
        "鹿谷鄉",
        "集集鎮",
    ],                
    "彰化縣": [
        "二水鄉",
        "二林鎮",
        "大村鄉",
        "大城鄉",
        "北斗鎮",
        "永靖鄉",
        "田中鎮",
        "田尾鄉",
        "竹塘鄉",
        "伸港鄉",
        "秀水鄉",
        "和美鎮",
        "社頭鄉",
        "芬園鄉",
        "花壇鄉",
        "芳苑鄉",
        "員林市",
        "埔心鄉",
        "埔鹽鄉",
        "埤頭鄉",
        "鹿港鎮",
        "溪州鄉",
        "溪湖鎮",
        "彰化市",
        "福興鄉",
        "線西鄉",
    ],           
    "雲林縣": [
        "二崙鄉",
        "口湖鄉",
        "土庫鎮",
        "大埤鄉",
        "元長鄉",
        "斗六市",
        "斗南鎮",
        "水林鄉",
        "北港鎮",
        "古坑鄉",
        "四湖鄉",
        "西螺鎮",
        "東勢鄉",
        "林內鄉",
        "虎尾鎮",
        "崙背鄉",
        "麥寮鄉",
        "莿桐鄉",
        "臺西鄉",
        "褒忠鄉",
    ],           
    "嘉義縣": [
        "大林鎮",
        "大埔鄉",
        "中埔鄉",
        "六腳鄉",
        "太保市",
        "水上鄉",
        "布袋鎮",
        "民雄鄉",
        "朴子市",
        "竹崎鄉",
        "東石鄉",
        "阿里山鄉",
        "梅山鄉",
        "鹿草鄉",
        "番路鄉",
        "新港鄉",
        "溪口鄉",
        "義竹鄉",
    ],            
    "臺南市": [
        "七股區",
        "下營區",
        "大內區",
        "山上區",
        "中西區",
        "仁德區",
        "六甲區",
        "北門區",
        "北區",
        "左鎮區",
        "永康區",
        "玉井區",
        "白河區",
        "安平區",
        "安定區",
        "安南區",
        "西港區",
        "佳里區",
        "官田區",
        "東山區",
        "東區",
        "南化區",
        "南區",
        "後壁區",
        "柳營區",
        "將軍區",
        "麻豆區",
        "善化區",
        "新化區",
        "新市區",
        "新營區",
        "楠西區",
        "學甲區",
        "龍崎區",
        "歸仁區",
        "關廟區",
        "鹽水區",
    ],            
    "高雄市": [
        "三民區",
        "大社區",
        "大寮區",
        "大樹區",
        "小港區",
        "仁武區",
        "內門區",
        "六龜區",
        "左營區",
        "永安區",
        "田寮區",
        "甲仙區",
        "杉林區",
        "那瑪夏區",
        "岡山區",
        "林園區",
        "阿蓮區",
        "前金區",
        "前鎮區",
        "美濃區",
        "苓雅區",
        "茂林區",
        "茄萣區",
        "桃源區",
        "梓官區",
        "鳥松區",
        "湖內區",
        "新興區",
        "楠梓區",
        "路竹區",
        "鼓山區",
        "旗山區",
        "旗津區",
        "鳳山區",
        "橋頭區",
        "燕巢區",
        "彌陀區",
        "鹽埕區",
    ],                
    "屏東縣": [
        "九如鄉",
        "三地門鄉",
        "內埔鄉",
        "竹田鄉",
        "牡丹鄉",
        "車城鄉",
        "里港鄉",
        "佳冬鄉",
        "來義鄉",
        "東港鎮",
        "枋山鄉",
        "枋寮鄉",
        "林邊鄉",
        "長治鄉",
        "南州鄉",
        "屏東市",
        "恆春鎮",
        "春日鄉",
        "崁頂鄉",
        "泰武鄉",
        "琉球鄉",
        "高樹鄉",
        "新埤鄉",
        "新園鄉",
        "獅子鄉",
        "萬丹鄉",
        "萬巒鄉",
        "滿州鄉",
        "瑪家鄉",
        "潮州鎮",
        "霧臺鄉",
        "麟洛鄉",
        "鹽埔鄉",
    ],       
    "宜蘭縣": [
        "三星鄉",
        "大同鄉",
        "五結鄉",
        "冬山鄉",
        "壯圍鄉",
        "宜蘭市",
        "南澳鄉",
        "員山鄉",
        "頭城鎮",
        "礁溪鄉",
        "羅東鎮",
        "蘇澳鎮",
    ],                    
    "花蓮縣": [
        "玉里鎮",
        "光復鄉",
        "吉安鄉",
        "秀林鄉",
        "卓溪鄉",
        "花蓮市",
        "富里鄉",
        "新城鄉",
        "瑞穗鄉",
        "萬榮鄉",
        "壽豐鄉",
        "鳳林鎮",
        "豐濱鄉",        
    ],                    
    "臺東縣": [
        "大武鄉",
        "太麻里鄉",
        "成功鎮",
        "池上鄉",
        "卑南鄉",
        "延平鄉",
        "東河鄉",
        "金峰鄉",
        "長濱鄉",
        "海端鄉",
        "鹿野鄉",
        "達仁鄉",
        "綠島鄉",
        "臺東市",
        "關山鎮",
        "蘭嶼鄉",
    ],          
    "基隆市": [
        "七堵區",
        "中山區",
        "中正區",
        "仁愛區",
        "安樂區",
        "信義區",
        "暖暖區",
    ],    
    "新竹市": [
        "北區",
        "東區",
        "香山區",        
    ],    
    "嘉義市": [
        "西區",
        "東區",
    ],  
    "澎湖縣": [
        "七美鄉",
        "白沙鄉",
        "西嶼鄉",
        "馬公市",
        "望安鄉",
        "湖西鄉",
    ],    
    "金門縣": [
        "金沙鎮",
        "金城鎮",
        "金湖鎮",
        "金寧鄉",
        "烈嶼鄉",
        "烏坵鄉",
    ],
    "連江縣": [
        "北竿鄉",
        "東引鄉",
        "南竿鄉",
        "莒光鄉",
    ],
    
}
# =========================
# 載入模型（快取）
# =========================
@st.cache_resource
def load_predictor():
    return HousePricePredictor()

predictor = load_predictor()

# =========================
# 側邊欄輸入
# =========================
st.sidebar.header("📋 房屋基本資料")

# --- 行政區（拆成兩欄） ---
city = st.sidebar.selectbox(
    "縣市",
    list(CITY_TOWN_MAP.keys()),
)

town = st.sidebar.selectbox(
    "鄉鎮市區",
    CITY_TOWN_MAP[city],
)

# ⭐ 合併成模型需要的格式
district = f"{city}{town}"
st.sidebar.caption(f"📍 行政區：{district}")

# --- 其他欄位 ---
building_type = st.sidebar.selectbox(
    "建物型態",
    ["住宅大樓", "華廈", "公寓", "透天厝"],
)

main_use = st.sidebar.selectbox(
    "主要用途",
    ["住家用", "商業用", "住商用"],
)

building_age = st.sidebar.number_input(
    "屋齡（年）",
    min_value=0,
    max_value=80,
    value=20,
)

main_area = st.sidebar.number_input(
    "主建物面積（坪）",
    min_value=5.0,
    max_value=100.0,
    value=30.0,
)

balcony_area = st.sidebar.number_input(
    "陽台面積（坪）",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
)

floor = st.sidebar.number_input(
    "所在樓層",
    min_value=1,
    max_value=100,
    value=5,
)

total_floors = st.sidebar.number_input(
    "總樓層數",
    min_value=1,
    max_value=100,
    value=10,
)

has_parking = st.sidebar.radio(
    "是否有車位",
    ["有", "無"],
)

has_elevator = st.sidebar.radio(
    "是否有電梯",
    ["有", "無"],
)

# =========================
# 組合輸入資料
# =========================
case_dict = {
    "district": district,
    "building_type": building_type,
    "main_use": main_use,
    "building_age": building_age,
    "building_area_sqm": main_area * 3.3058,
    "floor": floor,
    "total_floors": total_floors,
    "main_area": main_area,
    "balcony_area": balcony_area,
    "has_parking": 1 if has_parking == "有" else 0,
    "has_elevator": 1 if has_elevator == "有" else 0,
}

# =========================
# 主畫面
# =========================
st.subheader("📊 預測結果")

if st.button("🚀 開始估價"):

    with st.spinner("模型預測中，請稍候..."):
        output_dir = predictor.export_prediction_bundle(case_dict)

    # --- 讀取結果 ---
    with open(os.path.join(output_dir, "prediction.json"), encoding="utf-8") as f:
        summary = json.load(f)

    with open(os.path.join(output_dir, "explanation.txt"), encoding="utf-8") as f:
        explanation = f.read()

    # ====== 顯示預測價格 ======
    st.success(f"💰 預測單價：約 **{summary['predicted_price_wan_per_ping']} 萬 / 坪**")

    # ====== SHAP 圖 ======
    st.subheader("🔍 價格影響因素（SHAP）")
    st.image(
        os.path.join(output_dir, "shap_waterfall.png"),
        use_container_width=True,
    )

    # ====== 中文解釋 ======
    st.subheader("📝 中文估價說明")
    st.text(explanation)

    # ====== 下載區 ======
    st.subheader("⬇️ 下載結果")

    with open(os.path.join(output_dir, "explanation.txt"), "rb") as f:
        st.download_button(
            "📄 下載估價說明（TXT）",
            f,
            file_name="prediction.txt",
        )

    with open(os.path.join(output_dir, "shap_waterfall.png"), "rb") as f:
        st.download_button(
            "🖼️ 下載 SHAP 圖（PNG）",
            f,
            file_name="shap_waterfall.png",
        )

    with open(os.path.join(output_dir, "prediction.json"), "rb") as f:
        st.download_button(
            "📦 下載 JSON（API 用）",
            f,
            file_name="prediction.json",
        )

