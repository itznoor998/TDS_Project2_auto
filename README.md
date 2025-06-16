
# Autolysis 🧠📊

**Autolysis** is a Python-based automated data analysis tool designed to generate comprehensive statistical summaries, visualizations, and narrative reports from CSV datasets. It is built for data analysts, students, and professionals who want quick insights without manual scripting.

---

## 🚀 Features

- 📂 **Automatic data loading** from CSV files  
- 📊 **General statistics** (shape, missing values, unique categories, correlation matrix)  
- ⚠️ **Outlier detection** using Z-score method  
- 🔎 **Chi-Square tests** for categorical variable relationships  
- 📈 **Visualizations**:
  - Correlation heatmap
  - Histograms for numerical features
  - Pairplots of numerical relationships  
- ✍️ **Narrative generation** using LLM API (e.g., GPT)  
- 📝 Generates a complete **`README.md` report** with visualizations and insights

---

## 🛠 Dependencies

- Python >= 3.11  
- pandas  
- seaborn  
- matplotlib  
- scipy  
- requests  
- python-dotenv  

You can install them via:
```bash
pip install -r requirements.txt
```
*(Or manually install the packages listed above)*

---

## ⚙️ How to Run

```bash
uv run autolysis.py <path_to_your_dataset.csv>
```
✅ Example:
```bash
uv run autolysis.py data/sample.csv
```

---

## 📁 Output

- `README.md` — Contains statistical summary, narrative report, and embedded visualizations.
- `heatmap.png`, `histograms.png`, `pairplot.png` — Generated plots saved locally.

---

## 🌟 Highlights

- Modular design: Functions for loading data, computing stats, visualizing, and narrative generation  
- Environment variable support for secure API token management (.env file for `AIPROXY_TOKEN`)  
- Handles errors gracefully with clear messages  

---

## 🔐 API Usage

The narrative story is generated using an LLM API.  
✅ Ensure your `.env` file includes:
```
AIPROXY_TOKEN=your_api_token_here
```

---

## 📣 About

This project was developed to automate exploratory data analysis workflows and improve reporting efficiency.  
Feel free to contribute or fork for your own data projects!

---
