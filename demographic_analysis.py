import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")

plt.style.use("default")
sns.set_palette("husl")

# Единые интервалы для pd.cut по данным бюллетеня (возраст в полных годах)
BULLETIN_AGE_COARSE_BINS = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100]
BULLETIN_AGE_COARSE_LABELS = [
    "0-5",
    "6-10",
    "11-15",
    "16-20",
    "21-30",
    "31-40",
    "41-50",
    "51-60",
    "61-70",
    "71-80",
    "80+",
]
BULLETIN_AGE_FINE_BINS = [0, 15, 25, 35, 45, 55, 65, 75, 100]
BULLETIN_AGE_FINE_LABELS = [
    "0-14",
    "15-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65-74",
    "75+",
]

# Порядок возрастных групп в синтетических данных (должен совпадать с визуализацией)
SAMPLE_AGE_GROUPS = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85+",
]

BULLETIN_COLUMN_NAMES = [
    "Возраст",
    "Всего_все",
    "Всего_муж",
    "Всего_жен",
    "Город_все",
    "Город_муж",
    "Город_жен",
    "Село_все",
    "Село_муж",
    "Село_жен",
    "Содержание",
]


def _bulletin_data_start_row(df_raw: pd.DataFrame):
    """Первая строка таблицы с однолетним возрастом (0 или «до 1»)."""
    for i in range(len(df_raw)):
        c0 = str(df_raw.iat[i, 0]).strip().strip('"').replace("\n", " ")
        c0_lower = c0.lower()
        if not c0 or c0 == "nan":
            continue
        if c0_lower.startswith("до 1"):
            return i
        if c0 == "0":
            # Первая строка файла часто «0,1,2…» — отсекаем по численности в col1
            v1 = pd.to_numeric(df_raw.iat[i, 1], errors="coerce")
            if pd.notna(v1) and v1 > 100:
                return i
    return None


def _parse_bulletin_age_label(cell) -> float | None:
    """Преобразует метку возраста в число лет; None — строка не относится к однолетним возрастам."""
    s = str(cell).strip().strip('"').replace("\n", " ")
    if not s or s == "nan":
        return None
    s_lower = s.lower()
    if "трудоспособн" in s_lower:
        return None
    if "–" in s or "—" in s:
        return None
    if " и более" in s_lower or s_lower.endswith("и более"):
        return None
    if s_lower.startswith("всего") and len(s_lower) <= 8:
        return None
    if s_lower.startswith("до"):
        if "1" in s[:6]:
            return 0.0
        return None
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return None
    return float(v)


def _region_name_from_bulletin(path: str, df_raw: pd.DataFrame) -> str:
    base = os.path.basename(path).lower()
    if base.startswith("1_1_1"):
        return "Российская Федерация"
    for i in range(1, min(10, df_raw.shape[0])):
        name = str(df_raw.iat[i, 0]).strip().strip('"').replace("\n", " ")
        if not name or name == "nan":
            continue
        nl = name.lower()
        if "численность" in nl and "населения" in nl:
            continue
        if "человек" in nl or nl.startswith("1."):
            continue
        if "возраст" in nl:
            continue
        if len(name) > 3 and not name.replace(".", "").isdigit():
            return name
    return os.path.splitext(os.path.basename(path))[0]


def load_bulletin_table(filepath: str) -> tuple[str, pd.DataFrame]:
    """
    Загрузка одного CSV бюллетеня (РФ или субъект), единый формат колонок.
    Возвращает (название региона, DataFrame).
    """
    df_raw = pd.read_csv(filepath, encoding="utf-8-sig", header=None)
    start = _bulletin_data_start_row(df_raw)
    if start is None:
        raise ValueError(f"Не найдена таблица возрастов: {filepath}")
    data_rows = df_raw.iloc[start:].copy()
    ncols = data_rows.shape[1]
    data_rows.columns = BULLETIN_COLUMN_NAMES[:ncols]
    data_rows["Возраст"] = data_rows["Возраст"].map(_parse_bulletin_age_label)
    data_rows = data_rows.dropna(subset=["Возраст"])
    data_rows["Возраст"] = data_rows["Возраст"].astype(int)
    numeric_cols = [
        "Всего_все",
        "Всего_муж",
        "Всего_жен",
        "Город_все",
        "Город_муж",
        "Город_жен",
        "Село_все",
        "Село_муж",
        "Село_жен",
    ]
    present = [c for c in numeric_cols if c in data_rows.columns]
    data_rows[present] = (
        data_rows[present]
        .replace(["–", "—", "nan", ""], np.nan)
        .apply(lambda s: pd.to_numeric(s, errors="coerce"))
    )
    region = _region_name_from_bulletin(filepath, df_raw)
    return region, data_rows


def _is_bulletin_format(df):
    return df is not None and "Возраст" in df.columns and "Всего_все" in df.columns


class RussianDemographicAnalyzer:
    """
    Класс для анализа демографических данных России (Росстат)
    """

    def __init__(self):
        self.data = None
        self.regional_data: dict[str, pd.DataFrame] = {}

    def _load_regional_bulletins(self, data_dir: str = "data"):
        """Субъекты РФ: файлы вида data/2_1_1_.csv (не округа 2_1_.csv)."""
        self.regional_data.clear()
        root = Path(data_dir)
        if not root.is_dir():
            return
        for path in sorted(root.glob("2_*.csv")):
            stem_parts = [p for p in path.stem.split("_") if p]
            if len(stem_parts) < 3 or stem_parts[0] != "2":
                continue
            try:
                name, df = load_bulletin_table(str(path))
                if not _is_bulletin_format(df) or len(df) < 10:
                    continue
                self.regional_data[name] = df
            except (ValueError, OSError, UnicodeDecodeError):
                continue

    def load_rosstat_data(self, filename="data/russia_demography.csv"):
        """
        Загрузка данных из Росстат (формат CSV)
        Ожидаемые колонки: Возраст, Пол, Население, Год, Регион
        """
        bulletin_file = "data/1_1_1_.csv"

        if os.path.exists(bulletin_file):
            try:
                _, self.data = load_bulletin_table(bulletin_file)
                self._load_regional_bulletins(os.path.dirname(bulletin_file) or ".")
                print(
                    f"✅ Загружено {len(self.data):,} записей из бюллетеня Росстат (РФ)"
                )
                print(
                    f"📋 Возрастной диапазон: {self.data['Возраст'].min()} - {self.data['Возраст'].max()} лет"
                )
                if self.regional_data:
                    print(
                        f"🗺️ Загружено субъектов для сравнения: {len(self.regional_data)}"
                    )
                return self.data

            except Exception as e:
                print(f"⚠️ Ошибка загрузки бюллетеня: {e}")

        # Если файл не найден - используем старый метод
        try:
            self.data = pd.read_csv(filename, encoding="utf-8")
            print(f"✅ Загружено {len(self.data):,} записей из Росстат")
            print(f"📋 Доступные колонки: {', '.join(self.data.columns)}")

            # Базовая проверка данных
            print(f"📅 Годы: {self.data['Год'].min()} - {self.data['Год'].max()}")
            print(f"🗺️ Регионов: {self.data['Регион'].nunique()}")

        except FileNotFoundError:
            print(f"❌ Файл {filename} не найден")
            print(
                "💡 Скачайте данные с портала ЕМИСС: https://www.fedstat.ru/indicators/start.do"
            )
            self.generate_sample_data()
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            self.generate_sample_data()

        return self.data

    def generate_sample_data(self):
        """
        Демо-данные на основе структуры Росстат
        для тестирования без реальных данных
        """
        print("\n⚠️ Генерация тестовых данных на основе структуры Росстат...")

        regions = [
            "Москва",
            "Московская обл.",
            "Санкт-Петербург",
            "Ленинградская обл.",
            "Краснодарский край",
            "Свердловская обл.",
            "Татарстан",
            "Башкортостан",
        ]

        data_rows = []
        np.random.seed(42)

        for year in [2020, 2021, 2022, 2023]:
            for region in regions:
                # Базовое население региона
                base_pop = np.random.randint(500000, 5000000)

                for age_group in SAMPLE_AGE_GROUPS:
                    for gender in ["Мужской", "Женский"]:
                        # Распределение населения по возрастам
                        age_factor = {
                            "0-4": 0.05,
                            "5-9": 0.05,
                            "10-14": 0.05,
                            "15-19": 0.04,
                            "20-24": 0.04,
                            "25-29": 0.06,
                            "30-34": 0.07,
                            "35-39": 0.07,
                            "40-44": 0.07,
                            "45-49": 0.07,
                            "50-54": 0.07,
                            "55-59": 0.07,
                            "60-64": 0.07,
                            "65-69": 0.06,
                            "70-74": 0.05,
                            "75-79": 0.04,
                            "80-84": 0.03,
                            "85+": 0.02,
                        }.get(age_group, 0.05)

                        population = int(
                            base_pop
                            * age_factor
                            * (0.52 if gender == "Женский" else 0.48)
                        )

                        # Средний доход по региону (тыс. руб)
                        avg_income = np.random.normal(45, 10) * (
                            1.2 if region in ["Москва", "Санкт-Петербург"] else 1.0
                        )

                        data_rows.append(
                            {
                                "Год": year,
                                "Регион": region,
                                "Возрастная_группа": age_group,
                                "Пол": gender,
                                "Население": population,
                                "Средний_доход": round(avg_income, 1),
                                "Городское": np.random.choice([0, 1], p=[0.25, 0.75]),
                            }
                        )

        self.data = pd.DataFrame(data_rows)
        print(f"✅ Сгенерировано {len(self.data):,} записей")

    def basic_statistics(self):
        """
        Ключевая статистика по России
        """
        print("\n" + "=" * 60)
        print("🇷🇺 ОСНОВНЫЕ ДЕМОГРАФИЧЕСКИЕ ПОКАЗАТЕЛИ РОССИИ")
        print("=" * 60)

        if _is_bulletin_format(self.data):
            # Новый формат из бюллетеня
            total_pop = self.data["Всего_все"].sum() / 1_000_000
            print(f"\n👥 Общая численность населения: {total_pop:.2f} млн чел.")

            # Соотношение полов
            total_men = self.data["Всего_муж"].sum()
            total_women = self.data["Всего_жен"].sum()
            total = total_men + total_women
            print(f"\n👫 Соотношение полов:")
            print(f"   Мужчины: {total_men/total*100:.1f}%")
            print(f"   Женщины: {total_women/total*100:.1f}%")

            # Город/село
            urban = self.data["Город_все"].sum()
            rural = self.data["Село_все"].sum()
            urban_pct = urban / (urban + rural) * 100
            print(f"\n🏙️ Урбанизация: {urban_pct:.1f}% городского населения")
        else:
            # Старый формат (сгенерированные данные)
            total_pop = self.data["Население"].sum() / 1_000_000
            print(f"\n👥 Общая численность населения: {total_pop:.2f} млн чел.")

            # Соотношение полов
            gender_ratio = self.data.groupby("Пол")["Население"].sum()
            total = gender_ratio.sum()
            print(f"\n👫 Соотношение полов:")
            print(f"   Мужчины: {gender_ratio.get('Мужской', 0)/total*100:.1f}%")
            print(f"   Женщины: {gender_ratio.get('Женский', 0)/total*100:.1f}%")

            # Город/село
            urban = self.data.groupby("Городское")["Население"].sum()
            urban_pct = urban.get(1, 0) / total * 100
            print(f"\n🏙️ Урбанизация: {urban_pct:.1f}% городского населения")

            # Топ-5 регионов
            print(f"\n📊 КРУПНЕЙШИЕ РЕГИОНЫ (по населению):")
            region_pop = (
                self.data.groupby("Регион")["Население"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            for region, pop in region_pop.items():
                print(f"   {region}: {pop/1_000_000:.2f} млн чел.")

    def age_structure(self):
        """
        Возрастная структура населения
        """
        print("\n" + "=" * 60)
        print("📊 ВОЗРАСТНАЯ СТРУКТУРА")
        print("=" * 60)

        if _is_bulletin_format(self.data):
            total_pop = self.data["Всего_все"].sum()
            age = self.data["Возраст"]
            # Трудоспособный возраст (РФ): мужчины 16–59, женщины 16–54
            working = (
                self.data.loc[(age >= 16) & (age <= 59), "Всего_муж"].sum()
                + self.data.loc[(age >= 16) & (age <= 54), "Всего_жен"].sum()
            )

            young = self.data.loc[age < 16, "Всего_все"].sum()

            # Старше трудоспособного
            old = total_pop - working - young

            print(f"\n👶 Моложе трудоспособного возраста: {young/total_pop*100:.1f}%")
            print(f"👨‍💼 Трудоспособный возраст: {working/total_pop*100:.1f}%")
            print(f"👴 Старше трудоспособного возраста: {old/total_pop*100:.1f}%")

            # Демографическая нагрузка
            dep_ratio = (young + old) / working * 100 if working > 0 else 0
            print(f"\n⚖️ Коэффициент демографической нагрузки: {dep_ratio:.1f}%")
            print(
                f"   (на 1000 трудоспособных приходится {dep_ratio*10:.0f} нетрудоспособных)"
            )
        else:
            # Старый формат (сгенерированные данные)
            working_age = [
                "15-19",
                "20-24",
                "25-29",
                "30-34",
                "35-39",
                "40-44",
                "45-49",
                "50-54",
                "55-59",
            ]
            younger = ["0-4", "5-9", "10-14"]
            older = ["60-64", "65-69", "70-74", "75-79", "80-84", "85+"]

            def get_age_group_pop(groups):
                mask = self.data["Возрастная_группа"].isin(groups)
                return self.data[mask]["Население"].sum()

            total = self.data["Население"].sum()
            working = get_age_group_pop(working_age)
            young = get_age_group_pop(younger)
            old = get_age_group_pop(older)

            print(f"\n👶 Моложе трудоспособного возраста: {young/total*100:.1f}%")
            print(f"👨‍💼 Трудоспособный возраст: {working/total*100:.1f}%")
            print(f"👴 Старше трудоспособного возраста: {old/total*100:.1f}%")

            dep_ratio = (young + old) / working * 100 if working > 0 else 0
            print(f"\n⚖️ Коэффициент демографической нагрузки: {dep_ratio:.1f}%")
            print(
                f"   (на 1000 трудоспособных приходится {dep_ratio*10:.0f} нетрудоспособных)"
            )

    def regional_comparison(self):
        """
        Сравнение регионов или возрастных групп
        """
        print("\n" + "=" * 60)

        if _is_bulletin_format(self.data):
            # Новый формат - показываем распределение по возрастам
            print("📊 РАСПРЕДЕЛЕНИЕ НАСЕЛЕНИЯ ПО ВОЗРАСТАМ")
            print("=" * 60)

            df = self.data.copy()
            df["Возрастная_группа"] = pd.cut(
                df["Возраст"],
                bins=BULLETIN_AGE_COARSE_BINS,
                labels=BULLETIN_AGE_COARSE_LABELS,
            )

            age_dist = df.groupby("Возрастная_группа", observed=True)["Всего_все"].sum()
            total = age_dist.sum()

            print("\n👥 Распределение по возрастным группам:")
            for age, pop in age_dist.items():
                pct = pop / total * 100
                bar = "█" * int(pct / 2)
                print(f"   {age:>6}: {pop/1_000_000:>6.2f} млн ({pct:>5.1f}%) {bar}")
        else:
            # Старый формат
            print("🗺️ СРАВНЕНИЕ РЕГИОНОВ")
            print("=" * 60)

            # Агрегация по регионам
            regions = (
                self.data.groupby("Регион")
                .agg(
                    {
                        "Население": "sum",
                        "Средний_доход": "mean",
                        "Городское": lambda x: (x == 1).mean() * 100,
                    }
                )
                .round(1)
            )

            regions.columns = ["Население", "Ср. доход", "% городского"]
            regions = regions.sort_values("Население", ascending=False)

            print("\n🏆 ТОП-5 ПО НАСЕЛЕНИЮ:")
            print(regions.head(5).to_string())

            print("\n💰 ТОП-5 ПО ДОХОДУ:")
            print(
                regions.nlargest(5, "Ср. доход")[["Ср. доход", "Население"]].to_string()
            )

    def visualize(self):
        """
        Визуализация данных
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if _is_bulletin_format(self.data):
            # Новый формат из бюллетеня
            ax1 = axes[0]

            # Половозрастная пирамида
            df = self.data.sort_values("Возраст")
            ages = df["Возраст"].values
            men = df["Всего_муж"].values / 1_000_000
            women = df["Всего_жен"].values / 1_000_000

            y_pos = np.arange(len(ages))
            ax1.barh(y_pos, -men, color="blue", alpha=0.6, label="Мужчины")
            ax1.barh(y_pos, women, color="red", alpha=0.6, label="Женщины")

            ax1.set_yticks(y_pos[::5])
            ax1.set_yticklabels(ages[::5])
            ax1.set_xlabel("Население (млн чел.)")
            ax1.set_title("Половозрастная пирамида")
            ax1.legend()
            ax1.axvline(0, color="black", linewidth=0.5)

            # 2. Распределение по возрасту
            ax2 = axes[1]
            df["Возрастная_группа"] = pd.cut(
                df["Возраст"],
                bins=BULLETIN_AGE_FINE_BINS,
                labels=BULLETIN_AGE_FINE_LABELS,
            )
            age_dist = (
                df.groupby("Возрастная_группа", observed=True)["Всего_все"].sum()
                / 1_000_000
            )
            age_dist.plot(kind="bar", ax=ax2, color="green", alpha=0.7)
            ax2.set_xlabel("Возрастная группа")
            ax2.set_ylabel("Население (млн чел.)")
            ax2.set_title("Распределение по возрастным группам")
            ax2.tick_params(axis="x", rotation=45)
        else:
            # Старый формат
            ax1 = axes[0]

            # Агрегация по возрастным группам
            age_groups = (
                self.data.groupby(["Возрастная_группа", "Пол"])["Население"]
                .sum()
                .unstack()
            )

            age_groups = age_groups.reindex(SAMPLE_AGE_GROUPS)

            # Пирамида
            y_pos = np.arange(len(age_groups))

            ax1.barh(
                y_pos,
                -age_groups["Мужской"].values / 1_000_000,
                color="blue",
                alpha=0.6,
                label="Мужчины",
            )
            ax1.barh(
                y_pos,
                age_groups["Женский"].values / 1_000_000,
                color="red",
                alpha=0.6,
                label="Женщины",
            )

            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(age_groups.index)
            ax1.set_xlabel("Население (млн чел.)")
            ax1.set_title("Половозрастная пирамида")
            ax1.legend()
            ax1.axvline(0, color="black", linewidth=0.5)

            # 2. Доход по регионам (топ-10)
            ax2 = axes[1]

            region_income = (
                self.data.groupby("Регион")["Средний_доход"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            region_income.plot(kind="barh", ax=ax2, color="green", alpha=0.7)
            ax2.set_xlabel("Средний доход (тыс. руб)")
            ax2.set_title("Топ-10 регионов по доходу")

        plt.tight_layout()
        plt.savefig("russia_demography.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("\n✅ График сохранен в 'russia_demography.png'")


def main(use_gui=True):
    """
    Запуск анализа
    """
    print("🇷🇺 АНАЛИЗ ДЕМОГРАФИЧЕСКИХ ДАННЫХ РОССИИ")
    print("=" * 60)

    analyzer = RussianDemographicAnalyzer()

    # Загрузка данных (укажите путь к вашему файлу)
    data_file = "data/russia_demography.csv"  # замените на ваш файл

    analyzer.load_rosstat_data(data_file)

    if analyzer.data is not None:
        if use_gui:
            # Запуск GUI
            app = DemographicGUI(analyzer)
            app.run()
        else:
            # Только терминал
            analyzer.basic_statistics()
            analyzer.age_structure()
            analyzer.regional_comparison()
            analyzer.visualize()

            print("\n" + "=" * 60)
            print("✅ Анализ завершен")
            print("📊 Источник: Росстат / ЕМИСС (fedstat.ru)")
            print("ℹ️ Подробнее: rosstat.gov.ru")


class DemographicGUI:
    """Интерфейс анализа с выбором территории (РФ или субъект)."""

    THEME = {
        "app_bg": "#eef1f8",
        "header": "#0a1628",
        "header_accent": "#2563eb",
        "panel": "#ffffff",
        "panel_line": "#c8d4e6",
        "text": "#0f172a",
        "muted": "#64748b",
        "card": "#fafbfe",
        "card_line": "#dce4f0",
        "accent": "#2563eb",
        "accent_light": "#eff6ff",
        "field_bg": "#f8fafc",
    }

    def _font(self, size, weight="normal"):
        if sys.platform == "win32":
            return ("Segoe UI", size, weight)
        return ("Helvetica", size, weight)

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("Демография России · Росстат")
        self.root.geometry("1260x840")
        self.root.minsize(1040, 700)
        T = self.THEME
        self.root.configure(bg=T["app_bg"])

        self._rf_label = "Российская Федерация"
        self.region_var = tk.StringVar(value=self._rf_label)
        self._all_region_values: list[str] = []
        self.region_combo = None
        self.region_listbox = None
        self._region_hits: list[str] = []
        # Чтобы <<ComboboxSelected>> не пересобирал вкладки, если регион фактически тот же
        self._last_region_selection: str = self._rf_label

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background=T["app_bg"], foreground=T["text"])
        self.style.configure("TNotebook", background=T["app_bg"], borderwidth=0)
        self.style.configure(
            "TNotebook.Tab",
            background="#e2e8f0",
            foreground=T["text"],
            padding=[20, 10],
            font=self._font(10),
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", T["accent"]), ("active", "#cbd5e1")],
            foreground=[("selected", "#ffffff"), ("!selected", T["text"])],
        )
        self.style.configure("TFrame", background=T["app_bg"])
        self.style.configure("Card.TFrame", background=T["app_bg"])
        self.style.configure(
            "TEntry",
            fieldbackground=T["field_bg"],
            borderwidth=1,
            relief="flat",
            padding=(10, 8),
            font=self._font(10),
        )
        self.style.configure(
            "TCombobox",
            fieldbackground=T["field_bg"],
            borderwidth=1,
            padding=(8, 6),
            arrowsize=14,
            font=self._font(10),
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", T["field_bg"])],
            selectbackground=[("readonly", T["field_bg"])],
            selectforeground=[("readonly", T["text"])],
        )
        self.style.configure(
            "Treeview",
            font=self._font(10),
            rowheight=28,
            background=T["panel"],
            fieldbackground=T["panel"],
            borderwidth=0,
        )
        self.style.configure(
            "Treeview.Heading",
            font=self._font(10, "bold"),
            background=T["accent_light"],
            foreground=T["text"],
            relief="flat",
            borderwidth=0,
        )
        self.style.map(
            "Treeview",
            background=[("selected", T["accent"])],
            foreground=[("selected", "#ffffff"), ("!selected", T["text"])],
        )

        self.create_widgets()

    def _get_view_data(self):
        choice = self.region_var.get()
        if choice == self._rf_label or not choice:
            return self.analyzer.data
        return self.analyzer.regional_data.get(choice, self.analyzer.data)

    def _clear_tab(self, tab):
        for w in tab.winfo_children():
            w.destroy()

    def _on_region_change(self, _evt=None):
        new = self.region_var.get()
        if new == self._last_region_selection:
            return
        self._last_region_selection = new
        self._clear_tab(self.tab_main)
        self._clear_tab(self.tab_age)
        self._clear_tab(self.tab_viz)
        self._build_main_tab()
        self._build_age_tab()
        self._build_viz_tab()

    def _pop_unit(self, total_people: float) -> tuple[float, str, str]:
        if total_people >= 1_000_000:
            return 1_000_000.0, "млн", "млн чел."
        return 1_000.0, "тыс.", "тыс. чел."

    def _on_search_escape(self, _evt=None):
        self.region_var.set(
            self._all_region_values[0] if self._all_region_values else ""
        )
        self._apply_region_list_filter()
        self._open_region_combo()
        return "break"

    def _on_region_search(self, _evt=None):
        """Обработка ввода в поле Территория - фильтрация списка."""
        self._apply_region_list_filter(show_all_on_empty=False)
        if self._region_hits:
            self._open_region_combo()

    def _open_region_combo(self):
        """Показывает список регионов под полем ввода."""
        if self.region_listbox is None:
            return
        self.region_listbox.grid()

    def _hide_region_list(self):
        if self.region_listbox is not None:
            self.region_listbox.grid_remove()

    def _on_combo_click(self, _evt=None):
        """По клику в любую часть поля сразу показывает список."""
        self._apply_region_list_filter(show_all_on_empty=True, keep_text=True)
        self._open_region_combo()

    def _show_full_region_list(self, _evt=None):
        """Показать полный список регионов по нажатию ПКМ."""
        if self.region_combo is None:
            return
        self.region_var.set("")
        self._apply_region_list_filter(show_all_on_empty=True, keep_text=True)
        self._open_region_combo()

    def _on_combo_focus_in(self, _evt=None):
        """При фокусе на комбобоксе показать полный список."""
        if self.region_combo is None:
            return
        q = self.region_var.get().strip().casefold()
        if not q:
            self._apply_region_list_filter(show_all_on_empty=True, keep_text=False)
            self._open_region_combo()

    def _apply_region_list_filter(
        self, *_args, show_all_on_empty=True, keep_text=True
    ):
        """Сужает список по подстроке без подмены введенного текста."""
        if self.region_combo is None:
            return
        full = self._all_region_values
        if not full:
            return
        cur = self.region_var.get()
        q = cur.strip().casefold()
        if not q and show_all_on_empty:
            hits = full
        else:
            hits = [n for n in full if q in n.casefold()]
        self._region_hits = hits
        if self.region_listbox is not None:
            self.region_listbox.delete(0, tk.END)
            for name in hits:
                self.region_listbox.insert(tk.END, name)
            if hits:
                self.region_listbox.selection_clear(0, tk.END)
                self.region_listbox.selection_set(0)
                self.region_listbox.activate(0)
                self._open_region_combo()
            else:
                self._hide_region_list()
        if keep_text:
            self.region_var.set(cur)

    def _on_region_pick(self, _evt=None):
        """Выбор региона из списка под строкой ввода."""
        if self.region_listbox is None:
            return
        sel = self.region_listbox.curselection()
        if not sel:
            return
        picked = self.region_listbox.get(sel[0])
        self.region_var.set(picked)
        self._hide_region_list()
        self._on_region_change()

    def _on_region_enter(self, _evt=None):
        """Enter: выбрать первый подходящий регион или точное совпадение."""
        cur = self.region_var.get().strip()
        if not cur:
            return "break"
        exact = next((x for x in self._all_region_values if x.casefold() == cur.casefold()), None)
        if exact is not None:
            self.region_var.set(exact)
            self._hide_region_list()
            self._on_region_change()
            return "break"
        if self._region_hits:
            self.region_var.set(self._region_hits[0])
            self._hide_region_list()
            self._on_region_change()
            return "break"
        return "break"

    def create_widgets(self):
        T = self.THEME
        header = tk.Frame(self.root, bg=T["header"], height=96)
        header.pack(fill="x")
        header.pack_propagate(False)
        stripe = tk.Frame(header, bg=T["header_accent"], height=3)
        stripe.pack(side="bottom", fill="x")

        h_inner = tk.Frame(header, bg=T["header"])
        h_inner.pack(expand=True, fill="both", padx=32, pady=(20, 16))
        tk.Label(
            h_inner,
            text="Демографический профиль",
            font=self._font(24, "bold"),
            fg="#f8fafc",
            bg=T["header"],
        ).pack(anchor="w")
        tk.Label(
            h_inner,
            text="Половозрастная структура · город и село · 1 января 2024 г.",
            font=self._font(11),
            fg="#94a3b8",
            bg=T["header"],
        ).pack(anchor="w", pady=(8, 0))

        self._all_region_values = [self._rf_label]
        if getattr(self.analyzer, "regional_data", None):
            self._all_region_values.extend(
                sorted(self.analyzer.regional_data.keys(), key=str.casefold)
            )

        bar_outer = tk.Frame(self.root, bg=T["app_bg"])
        bar_outer.pack(fill="x", padx=24, pady=(18, 0))
        bar = tk.Frame(
            bar_outer,
            bg=T["panel"],
            highlightthickness=1,
            highlightbackground=T["panel_line"],
        )
        bar.pack(fill="x")
        bar_in = tk.Frame(bar, bg=T["panel"])
        bar_in.pack(fill="x", padx=22, pady=(18, 18))
        bar_in.grid_columnconfigure(1, weight=0)

        territory_label = tk.Label(
            bar_in,
            text="Территория",
            font=self._font(10, "bold"),
            fg=T["text"],
            bg=T["panel"],
        )
        territory_label.grid(row=0, column=0, sticky="ne", padx=(0, 14), pady=(4, 0))
        self.region_combo = ttk.Entry(
            bar_in,
            textvariable=self.region_var,
            width=44,
        )
        self.region_combo.grid(row=0, column=1, sticky="w", pady=(0, 2))
        self.region_combo.bind("<KeyRelease>", self._on_region_search)
        self.region_combo.bind("<Escape>", self._on_search_escape)
        self.region_combo.bind("<Return>", self._on_region_enter)
        self.region_combo.bind("<Button-1>", self._on_combo_click, add="+")
        self.region_combo.bind("<Button-3>", lambda e: self._show_full_region_list())
        self.region_combo.bind("<FocusIn>", self._on_combo_focus_in)
        territory_label.bind("<Button-1>", self._on_combo_click)
        self.region_listbox = tk.Listbox(
            bar_in,
            height=8,
            width=44,
            bg=T["field_bg"],
            fg=T["text"],
            activestyle="dotbox",
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            font=self._font(10),
        )
        self.region_listbox.grid(row=1, column=1, sticky="w", pady=(0, 8))
        self.region_listbox.bind("<ButtonRelease-1>", self._on_region_pick)
        self.region_listbox.bind("<Return>", self._on_region_pick)
        self.region_listbox.grid_remove()
        tk.Label(
            bar_in,
            text="Введите название — список сузится. Esc — очистить. ПКМ — полный список.",
            font=self._font(9),
            fg=T["muted"],
            bg=T["panel"],
        ).grid(row=2, column=1, sticky="w", pady=(0, 12))

        body = tk.Frame(self.root, bg=T["app_bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(16, 22))

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill="both", expand=True)

        self.tab_main = ttk.Frame(self.notebook, padding=0, style="Card.TFrame")
        self.tab_age = ttk.Frame(self.notebook, padding=0, style="Card.TFrame")
        self.tab_viz = ttk.Frame(self.notebook, padding=0, style="Card.TFrame")
        self.tab_source = ttk.Frame(self.notebook, padding=0, style="Card.TFrame")

        self.notebook.add(self.tab_main, text="Основные")
        self.notebook.add(self.tab_age, text="Возраст")
        self.notebook.add(self.tab_viz, text="Графики")
        self.notebook.add(self.tab_source, text="Источник")

        self._build_main_tab()
        self._build_age_tab()
        self._build_viz_tab()
        self._build_source_tab()

    def _build_main_tab(self):
        T = self.THEME
        data = self._get_view_data()
        if data is None or not _is_bulletin_format(data):
            tk.Label(
                self.tab_main,
                text="Нет данных в формате бюллетеня Росстат.",
                font=self._font(12),
                bg=T["app_bg"],
                fg=T["muted"],
            ).pack(pady=40)
            return

        total_all = float(data["Всего_все"].sum())
        div, short_unit, long_unit = self._pop_unit(total_all)
        total_pop = total_all / div

        total_men = float(data["Всего_муж"].sum())
        total_women = float(data["Всего_жен"].sum())
        denom = total_men + total_women
        men_pct = (total_men / denom * 100) if denom else 0.0
        women_pct = (total_women / denom * 100) if denom else 0.0

        urban = float(data["Город_все"].fillna(0).sum())
        rural = float(data["Село_все"].fillna(0).sum())
        ur_sum = urban + rural
        urban_pct = (urban / ur_sum * 100) if ur_sum > 0 else 0.0

        shell = tk.Frame(self.tab_main, bg=T["app_bg"])
        shell.pack(fill="both", expand=True, padx=12, pady=12)

        title = self.region_var.get()
        panel = tk.Frame(
            shell,
            bg=T["panel"],
            highlightthickness=1,
            highlightbackground=T["panel_line"],
        )
        panel.pack(fill="both", expand=True)

        tk.Label(
            panel,
            text=title,
            font=self._font(17, "bold"),
            fg=T["text"],
            bg=T["panel"],
        ).pack(anchor="w", padx=22, pady=(20, 4))
        tk.Label(
            panel,
            text="Ключевые показатели по выбранной территории",
            font=self._font(10),
            fg=T["muted"],
            bg=T["panel"],
        ).pack(anchor="w", padx=22, pady=(0, 16))

        cards = tk.Frame(panel, bg=T["panel"])
        cards.pack(fill="x", padx=16, pady=(0, 8))
        for c in range(3):
            cards.columnconfigure(c, weight=1, uniform="metric")
        cards.rowconfigure(0, weight=1)
        cards.rowconfigure(1, weight=1)

        pop_label = (
            f"{total_pop:.2f} {short_unit} чел."
            if div >= 1_000_000
            else f"{total_pop:.1f} {short_unit} чел."
        )
        self._metric_card(cards, 0, "Население", pop_label, long_unit)
        self._metric_card(
            cards, 1, "Мужчины", f"{men_pct:.1f}%", "доля от половой структуры"
        )
        self._metric_card(
            cards, 2, "Женщины", f"{women_pct:.1f}%", "доля от половой структуры"
        )
        self._metric_card(
            cards, 3, "Город", f"{urban_pct:.1f}%", "доля городского населения"
        )
        self._metric_card(
            cards, 4, "Село", f"{100 - urban_pct:.1f}%", "доля сельского населения"
        )
        age_min = int(data["Возраст"].min())
        age_max = int(data["Возраст"].max())
        self._metric_card(
            cards,
            5,
            "Возраст в таблице",
            f"{age_min} — {age_max} лет",
            "однолетние интервалы",
        )

        if self.region_var.get() == self._rf_label and getattr(
            self.analyzer, "regional_data", None
        ):
            rdata = self.analyzer.regional_data
            if rdata:
                ranked = [
                    (name, float(dfc["Всего_все"].sum()))
                    for name, dfc in rdata.items()
                    if dfc is not None
                    and "Всего_все" in dfc.columns
                    and _is_bulletin_format(dfc)
                ]
                ranked.sort(key=lambda x: -x[1])
                top10 = ranked[:10]
                if top10:
                    sep = ttk.Separator(panel, orient="horizontal")
                    sep.pack(fill="x", padx=16, pady=(12, 14))

                    top_block = tk.Frame(panel, bg=T["panel"])
                    top_block.pack(fill="both", expand=True, padx=16, pady=(0, 16))

                    tk.Label(
                        top_block,
                        text="Топ-10 субъектов РФ по численности",
                        font=self._font(13, "bold"),
                        fg=T["text"],
                        bg=T["panel"],
                    ).pack(anchor="w", pady=(0, 2))
                    tk.Label(
                        top_block,
                        text="По сумме «всё население» в загруженных бюллетенях по субъектам",
                        font=self._font(9),
                        fg=T["muted"],
                        bg=T["panel"],
                    ).pack(anchor="w", pady=(0, 10))

                    table_holder = tk.Frame(top_block, bg=T["panel"])
                    table_holder.pack(fill="both", expand=True)

                    cols = ("№", "Субъект", "Численность")
                    tree = ttk.Treeview(
                        table_holder,
                        columns=cols,
                        show="headings",
                        height=len(top10),
                        selectmode="browse",
                    )
                    tree.heading("№", text="№", anchor="center")
                    tree.column(
                        "№", width=44, minwidth=40, anchor="center", stretch=False
                    )
                    tree.heading("Субъект", text="Субъект", anchor="w")
                    tree.column(
                        "Субъект", width=520, minwidth=160, anchor="w", stretch=True
                    )
                    tree.heading("Численность", text="Численность", anchor="e")
                    tree.column(
                        "Численность",
                        width=130,
                        minwidth=100,
                        anchor="e",
                        stretch=False,
                    )

                    for i, (rname, ptot) in enumerate(top10, start=1):
                        ddiv, short_u, _ = self._pop_unit(ptot)
                        if ddiv >= 1_000_000:
                            val_s = f"{ptot / ddiv:.2f} {short_u} чел."
                        else:
                            val_s = f"{ptot / ddiv:.1f} {short_u} чел."
                        tree.insert("", "end", values=(i, rname, val_s))

                    vsb = ttk.Scrollbar(
                        table_holder, orient="vertical", command=tree.yview
                    )
                    tree.configure(yscrollcommand=vsb.set)
                    tree.pack(side="left", fill="both", expand=True)
                    vsb.pack(side="right", fill="y")

    def _metric_card(self, parent, index, title, value, hint):
        T = self.THEME
        row, col = divmod(index, 3)
        card = tk.Frame(
            parent,
            bg=T["card"],
            highlightthickness=1,
            highlightbackground=T["card_line"],
        )
        card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
        card.grid_propagate(False)
        card.configure(height=118)
        tk.Label(
            card,
            text=title.upper(),
            font=self._font(9),
            fg=T["muted"],
            bg=T["card"],
        ).pack(anchor="w", padx=14, pady=(12, 2))
        tk.Label(
            card,
            text=value,
            font=self._font(19, "bold"),
            fg=T["text"],
            bg=T["card"],
        ).pack(anchor="w", padx=14)
        tk.Label(
            card,
            text=hint,
            font=self._font(9),
            fg=T["muted"],
            bg=T["card"],
            wraplength=260,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(4, 12))

    def _build_age_tab(self):
        T = self.THEME
        data = self._get_view_data()
        if data is None or not _is_bulletin_format(data):
            tk.Label(
                self.tab_age,
                text="Нет данных.",
                font=self._font(12),
                bg=T["app_bg"],
                fg=T["muted"],
            ).pack(pady=40)
            return

        total_all = float(data["Всего_все"].sum())
        div, short_unit, _ = self._pop_unit(total_all)

        shell = tk.Frame(self.tab_age, bg=T["app_bg"])
        shell.pack(fill="both", expand=True, padx=12, pady=12)

        panel = tk.Frame(
            shell,
            bg=T["panel"],
            highlightthickness=1,
            highlightbackground=T["panel_line"],
        )
        panel.pack(fill="both", expand=True)

        tk.Label(
            panel,
            text=self.region_var.get(),
            font=self._font(16, "bold"),
            fg=T["text"],
            bg=T["panel"],
        ).pack(anchor="w", padx=20, pady=(18, 4))
        tk.Label(
            panel,
            text="Распределение по возрастным группам",
            font=self._font(10),
            fg=T["muted"],
            bg=T["panel"],
        ).pack(anchor="w", padx=20, pady=(0, 12))

        df = data.copy()
        df["Возрастная_группа"] = pd.cut(
            df["Возраст"],
            bins=BULLETIN_AGE_COARSE_BINS,
            labels=BULLETIN_AGE_COARSE_LABELS,
        )
        age_dist = df.groupby("Возрастная_группа", observed=True)["Всего_все"].sum()
        total = float(age_dist.sum())

        table_wrap = tk.Frame(panel, bg=T["panel"])
        table_wrap.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        col_pop = f"Числ. ({short_unit})"
        cols = ("Группа", col_pop, "Доля %", "Диаграмма")
        tree = ttk.Treeview(
            table_wrap, columns=cols, show="headings", height=14, selectmode="browse"
        )
        for col in cols:
            tree.heading(col, text=col)
        tree.column("Группа", width=118, minwidth=90, anchor="w", stretch=False)
        tree.column(col_pop, width=100, minwidth=80, anchor="e", stretch=False)
        tree.column("Доля %", width=72, minwidth=60, anchor="e", stretch=False)
        tree.column("Диаграмма", width=200, minwidth=120, anchor="w", stretch=True)

        for age, pop in age_dist.items():
            pct = (float(pop) / total * 100) if total else 0.0
            bar = "█" * max(0, min(40, int(pct / 2.5)))
            tree.insert(
                "",
                "end",
                values=(
                    str(age),
                    (
                        f"{float(pop) / div:.2f}"
                        if div >= 1_000_000
                        else f"{float(pop) / div:.1f}"
                    ),
                    f"{pct:.1f}",
                    bar,
                ),
            )

        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    def _build_viz_tab(self):
        T = self.THEME
        data = self._get_view_data()
        if data is None or not _is_bulletin_format(data):
            tk.Label(
                self.tab_viz,
                text="Нет данных для графиков.",
                font=self._font(12),
                bg=T["app_bg"],
                fg=T["muted"],
            ).pack(pady=40)
            return

        title = self.region_var.get()
        fig = Figure(figsize=(11, 9.0), facecolor=T["app_bg"])
        fig.patch.set_facecolor(T["app_bg"])

        men_raw = data["Всего_муж"].fillna(0).to_numpy()
        women_raw = data["Всего_жен"].fillna(0).to_numpy()
        max_side = float(np.max(np.r_[men_raw, women_raw])) if len(men_raw) else 0.0
        div, _, ylab = self._pop_unit(max(max_side, 1.0))

        ax1 = fig.add_subplot(2, 1, 1)
        df = data.sort_values("Возраст")
        ages = df["Возраст"].values
        men = df["Всего_муж"].fillna(0).values / div
        women = df["Всего_жен"].fillna(0).values / div
        y_pos = np.arange(len(ages))
        ax1.barh(y_pos, -men, color="#3b82f6", alpha=0.85, label="Мужчины", height=0.92)
        ax1.barh(
            y_pos, women, color="#e11d48", alpha=0.82, label="Женщины", height=0.92
        )
        step = max(1, len(ages) // 12)
        ax1.set_yticks(y_pos[::step])
        ax1.set_yticklabels(ages[::step], fontsize=9)
        ax1.set_xlabel(ylab, fontsize=10)
        ax1.set_title(
            f"Половозрастная структура — {title}",
            fontsize=13,
            fontweight="bold",
            color=T["text"],
            pad=12,
        )
        ax1.legend(loc="lower right", framealpha=0.95)
        ax1.axvline(0, color="#94a3b8", linewidth=0.6)
        ax1.grid(axis="x", alpha=0.25)
        ax1.set_facecolor("#f8fafc")

        ax2 = fig.add_subplot(2, 1, 2)
        d2 = df.copy()
        d2["Возрастная_группа"] = pd.cut(
            d2["Возраст"],
            bins=BULLETIN_AGE_FINE_BINS,
            labels=BULLETIN_AGE_FINE_LABELS,
        )
        age_dist = (
            d2.groupby("Возрастная_группа", observed=True)["Всего_все"].sum() / div
        )
        colors = [
            "#0ea5e9",
            "#6366f1",
            "#8b5cf6",
            "#db2777",
            "#f97316",
            "#10b981",
            "#64748b",
            "#14b8a6",
        ]
        x = np.arange(len(age_dist))
        bars = ax2.bar(
            x,
            age_dist.values,
            color=colors[: len(age_dist)],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(list(age_dist.index), rotation=35, ha="right", fontsize=9)
        ax2.set_ylabel(ylab, fontsize=10)
        ax2.set_title(
            "Крупные возрастные группы",
            fontsize=12,
            fontweight="bold",
            color=T["text"],
            pad=10,
        )
        ax2.grid(axis="y", alpha=0.25)
        ax2.set_facecolor("#f8fafc")
        ymax = float(np.max(age_dist.values)) if len(age_dist) else 0.0
        y_off = max(ymax * 0.03, 1e-6)
        for bar, val in zip(bars, age_dist.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_off,
                f"{val:.2f}" if div >= 1_000_000 else f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=T["text"],
            )

        fig.tight_layout(pad=1.8)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_viz)
        canvas.draw()
        canvas.get_tk_widget().configure(bg=T["app_bg"])
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _build_source_tab(self):
        T = self.THEME
        self._clear_tab(self.tab_source)
        outer = tk.Frame(self.tab_source, bg=T["app_bg"])
        outer.pack(fill="both", expand=True, padx=12, pady=12)

        panel = tk.Frame(
            outer,
            bg=T["panel"],
            highlightthickness=1,
            highlightbackground=T["panel_line"],
        )
        panel.pack(fill="both", expand=True)

        tk.Label(
            panel,
            text="Источник и методика",
            font=self._font(16, "bold"),
            fg=T["text"],
            bg=T["panel"],
        ).pack(anchor="w", padx=24, pady=(22, 8))

        nreg = len(getattr(self.analyzer, "regional_data", {}) or {})
        info = f"""Федеральная служба государственной статистики (Росстат).

Таблица 1.1.1. — численность населения Российской Федерации по полу и возрасту на 1 января 2024 г.
Таблицы 2.x.x — те же показатели по субъектам РФ (файлы в папке data/).

В интерфейсе можно переключаться между агрегатом по стране и отдельными субъектами.
Сейчас в каталоге загружено субъектов: {nreg}.

ЕМИСС: https://www.fedstat.ru/indicators/start.do
Сайт Росстата: https://rosstat.gov.ru
"""
        tw = scrolledtext.ScrolledText(
            panel,
            width=92,
            height=22,
            font=self._font(11),
            bg="#f1f5f9",
            fg=T["text"],
            relief="flat",
            padx=12,
            pady=12,
            wrap="word",
        )
        tw.insert("1.0", info)
        tw.configure(state="disabled")
        tw.pack(fill="both", expand=True, padx=20, pady=(0, 22))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    use_gui = "--no-gui" not in sys.argv
    main(use_gui)
