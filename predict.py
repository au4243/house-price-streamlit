"""
=================================
XGBoost 房價預測最終部署版模組（單檔完整版）

功能：
- 載入已訓練模型
- 單筆房價預測（萬 / 坪）
- 嚴格 one-hot 特徵對齊（高效、不碎片化）
- 單筆 SHAP 解釋（圖 + 中文文字）
- 輸出可下載檔案（txt / png / json）

使用方式：
from predict import HousePricePredictor
=================================
"""
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# Matplotlib 中文設定
# =========================

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False


class HousePricePredictor:
    """
    房價預測與 SHAP 解釋模組（正式部署等級）
    """

    def __init__(
        self,
        model_path: str = "xgb_house_price_model.pkl",
        feature_path: str = "model_features.pkl",
    ):
        # 載入模型與特徵
        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)

        # SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # 類別欄位（需與訓練時完全一致）
        self.categorical_cols = [
            "district",
            "building_type",
            "main_use",
        ]

    # --------------------------------------------------
    @staticmethod
    def _pretty_name(col: str, value=None) -> str:
        """
        將模型欄位轉為中文可讀名稱
        """
        if col.startswith("district_"):
            return f"行政區：{col.replace('district_', '')}"

        if col.startswith("building_type_"):
            return f"建物型態：{col.replace('building_type_', '')}"

        if col.startswith("main_use_"):
            return f"主要用途：{col.replace('main_use_', '')}"

        mapping = {
            "building_age": "屋齡（年）",
            "building_area_sqm": "建物移轉面積（㎡）",
            "main_area": "主建物面積（坪）",
            "balcony_area": "陽台面積（坪）",
            "floor": "所在樓層",
            "total_floors": "總樓層數",
            "has_parking": "是否有車位",
            "has_elevator": "是否有電梯",
        }

        name = mapping.get(col, col)
        return f"{name} = {value}" if value is not None else name

    # --------------------------------------------------
    def _preprocess(self, case_dict: dict) -> pd.DataFrame:
        """
        單筆資料 → 模型可接受格式
        - 高效補齊缺失欄位
        - 嚴格依照訓練特徵順序
        """
        df = pd.DataFrame([case_dict])

        # One-hot encoding
        df = pd.get_dummies(
            df,
            columns=self.categorical_cols,
            drop_first=False,
        )

        # 一次補齊所有缺失欄位（避免 DataFrame fragmentation）
        missing_cols = set(self.model_features) - set(df.columns)
        if missing_cols:
            missing_df = pd.DataFrame(
                0,
                index=df.index,
                columns=list(missing_cols),
            )
            df = pd.concat([df, missing_df], axis=1)

        # 依訓練時欄位順序排列
        df = df[self.model_features]

        return df

    # --------------------------------------------------
    def predict(self, case_dict: dict) -> float:
        """
        預測單筆房屋單價（萬 / 坪）
        """
        X_case = self._preprocess(case_dict)
        price = float(self.model.predict(X_case)[0])
        return price

    # --------------------------------------------------
    def shap_values(self, case_dict: dict):
        """
        回傳單筆 SHAP values 與對應特徵
        """
        X_case = self._preprocess(case_dict)
        shap_values = self.explainer.shap_values(X_case)
        return shap_values, X_case

    # --------------------------------------------------
    def generate_chinese_explanation(
        self,
        case_dict: dict,
        top_n: int = 8,
    ) -> str:
        """
        自動產生中文 SHAP 估價說明
        """
        shap_values, X_case = self.shap_values(case_dict)

        sv = shap_values[0]
        base = float(self.explainer.expected_value)
        pred = base + sv.sum()

        # 依影響力排序
        items = sorted(
            zip(X_case.columns, sv, X_case.iloc[0]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        lines = [
            f"本模型以整體樣本平均單價 {base:.2f} 萬 / 坪為基準，",
            f"此物件預測單價約為 {pred:.2f} 萬 / 坪。",
            "",
            "主要影響因素如下：",
        ]

        for col, shap_val, data in items:
            direction = "提高" if shap_val > 0 else "降低"
            lines.append(
                f"- {self._pretty_name(col, data)}，"
                f"使單價約{direction} {abs(shap_val):.2f} 萬 / 坪"
            )

        return "\n".join(lines)

    # --------------------------------------------------
    def plot_shap_waterfall(self, case_dict: dict):
        """
        繪製單筆 SHAP waterfall plot
        """
        shap_values, X_case = self.shap_values(case_dict)

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=float(self.explainer.expected_value),
                data=X_case.iloc[0],
                feature_names=X_case.columns,
            ),
            show=True,
        )


# ======================================================
# CLI / 本地測試入口
# ======================================================
if __name__ == "__main__":

    predictor = HousePricePredictor()

    sample_case = {
        "district": "臺北市內湖區",
        "building_type": "住宅大樓",
        "main_use": "住家用",
        "building_age": 55,
        "building_area_sqm": 45,
        "floor": 8,
        "total_floors": 15,
        "main_area": 30,
        "balcony_area": 5,
        "has_parking": 1,
        "has_elevator": 1,
    }

    # 預測
    price = predictor.predict(sample_case)
    print(f"\n預測單價：約 {price:.2f} 萬 / 坪\n")

    # 中文解釋
    explanation = predictor.generate_chinese_explanation(sample_case)
    print(explanation)

    # SHAP 圖
    predictor.plot_shap_waterfall(sample_case)



    case1 = {"building_type": "住宅大樓",
        "main_use": "住家用",
        "building_age": 55,
        "building_area_sqm": 45,
        "floor": 8,
        "total_floors": 15,
        "main_area": 30,
        "balcony_area": 5,
        "has_parking": 1,
        "has_elevator": 1, "district": "臺北市萬華區" }
    case2 = {"building_type": "住宅大樓",
        "main_use": "住家用",
        "building_age": 55,
        "building_area_sqm": 45,
        "floor": 8,
        "total_floors": 15,
        "main_area": 30,
        "balcony_area": 5,
        "has_parking": 1,
        "has_elevator": 1, "district": "臺北市大安區" }
    case3 = {"building_type": "住宅大樓",
        "main_use": "住家用",
        "building_age": 55,
        "building_area_sqm": 45,
        "floor": 8,
        "total_floors": 15,
        "main_area": 30,
        "balcony_area": 5,
        "has_parking": 1,
        "has_elevator": 1, "district": "臺北市內湖區" }

    print(predictor.predict(case1))
    print(predictor.predict(case2))
    print(predictor.predict(case3))
