# Gender Guesser Tool

A Python tool for predicting gender from first names using statistical data from Texas marriage records (1968-2015).

## Author
**Benji Stalvey** - *Creator and Developer*

## Dataset
- **300,000+ unique names** from **8+ million marriage records**
- **Source**: Texas marriage licenses (1968-2015)
- **Coverage**: Primarily US-English names 

## Installation

1. Download both files:
   - `Gender_Guessing_Tool.py`
   - `name_gender_stats.csv`

2. Install pandas:
   ```bash
   pip install pandas
   ```

3. Place both files in your project directory

## Quick Start

```python
import pandas as pd
from Gender_Guessing_Tool import predict_gender_for_dataframe

# Load your data
df = pd.read_csv('your_data.csv')

# Apply gender prediction
result = predict_gender_for_dataframe(
    df=df, 
    name_column='first_name',  # Replace with your column name that has the first name of the entry you would like to apply gender to
    output_file_path='results.csv'
)

# View results
print(result[['first_name', 'Percent_Male', 'Gender_Code']])
```

## Output Columns

| Column | Description |
|--------|-------------|
| `Percent_Male` | Percentage (0-100) of male usage for this name |
| `Gender_Code` | **M** = Male (>55%), **F** = Female (<45%), **A** = Ambiguous (45-55%), **U** = Unknown |
| `Found_In_Database` | True if name was found in the dataset |

## Examples

**Input:**
| first_name |
|------------|
| James      |
| Maria      |
| Alex       |
| Jordan     |

**Output:**
| first_name | Percent_Male | Gender_Code |
|------------|--------------|-------------|
| James      | 99.94         | M           |
| Maria      | 0.04         | F           |
| Ariel      | 53.38         | A           |
| Br!an     | NaN         | U           | 

## Usage with Different File Types

**CSV (Recommended):**
```python
df = pd.read_csv('data.csv')
result = predict_gender_for_dataframe(df, 'name_column', 'output.csv')
```

**Excel:**
```python
df = pd.read_excel('data.xlsx')
result = predict_gender_for_dataframe(df, 'name_column', 'output.csv')
```

**Any pandas-supported format:**
```python
df = pd.read_json('data.json')  # or read_parquet, etc.
result = predict_gender_for_dataframe(df, 'name_column', 'output.csv')
```

## Individual Name Lookup

```python
from Gender_Guessing_Tool import GenderGuesser

guesser = GenderGuesser()
prediction = guesser.predict_gender('Alex')
print(prediction)
# {'percent_male': 98.55, 'gender_code': 'M', 'found_in_database': True}
```

## Expected Results

- **Unknown Rate**: Higher for non-English names, nicknames, or unusual spellings
- **Performance**: Handles datasets with millions of records
- **Accuracy**: Based on historical frequency data from real marriage records

## File Recommendations

- **Use CSV** for output (faster, unlimited size)
- **Excel requires**: `pip install openpyxl`
- **Large datasets**: CSV strongly recommended over Excel

## Citation

If you use this tool in research or analysis, please cite:


Stalvey, B. (2025). Gender Guesser Tool. GitHub repository.
https://github.com/benjistalvey5/gender-guesser-tool

## License

This project is open source. Feel free to use, modify, and distribute.
