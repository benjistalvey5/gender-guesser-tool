"""
Gender Guessing Tool - A Python Package for Name-Based Gender Prediction

This package provides functionality to predict gender based on first names using 
statistical data derived from Texas marriage records (1968-2016).

Author: Benji Stalvey
Created for: Data Analysis Projects

================================================================================
STEP-BY-STEP USAGE GUIDE
================================================================================

STEP 1: SETUP AND INSTALLATION
------------------------------
1. Ensure you have pandas installed: pip install pandas
2. Place this file (Gender_Guessing_Tool.py) in your project directory
3. Place the name_gender_stats.csv file in the same directory
4. Optional: Install additional libraries for different file formats:
   - For Excel: pip install openpyxl
   - For Parquet: pip install pyarrow

STEP 2: BASIC USAGE (QUICK START)
---------------------------------
```python
import pandas as pd
from Gender_Guessing_Tool import predict_gender_for_dataframe

# Load your data
df = pd.read_csv('your_data.csv')

# Apply gender prediction
result = predict_gender_for_dataframe(
    df=df, 
    name_column='first_name',  # Replace with your column name
    output_file_path='results.csv'
)

# View results
print(result[['first_name', 'Percent_Male', 'Gender_Code']])
```

STEP 3: UNDERSTANDING THE OUTPUT
--------------------------------
Your DataFrame will get these new columns:
- 'Percent_Male': 0-100 percentage (50 = equal male/female usage)
- 'Gender_Code': 
  * "M" = Male (>55% male usage)
  * "F" = Female (<45% male usage)
  * "A" = Ambiguous (45-55% male usage)
  * "U" = Unknown (name not found in database)
- 'Found_In_Database': True/False if name was found

STEP 4: ADVANCED USAGE
----------------------
```python
from Gender_Guessing_Tool import GenderGuesser

# Create guesser object for multiple operations
guesser = GenderGuesser()

# Check individual names
prediction = guesser.predict_gender('Alex')
print(prediction)  # {'percent_male': 73.8, 'gender_code': 'M', 'found_in_database': True}

# Process DataFrame with custom settings
result = guesser.apply_gender_to_dataframe(
    df=your_data,
    name_column='customer_name',
    include_stats=True,
    output_file_path='output.csv'
)
```

STEP 5: TROUBLESHOOTING
-----------------------
- "Module not found": Make sure this file is in your current directory
- "File not found": Ensure name_gender_stats.csv is in the same folder
- "Column not found": Check your column name spelling and case sensitivity
- Excel errors: Install openpyxl or use CSV format instead
- Performance issues: Use CSV format for large datasets (>100k rows)

================================================================================
"""

import pandas as pd
import os
from typing import Optional, Union
import warnings
import sys


class GenderGuesser:
    """
    A class for predicting gender based on first names using statistical data.
    
    This tool uses a pre-compiled dataset of names with their male/female frequency
    statistics to predict gender for new datasets.
    """
    
    def __init__(self, data_file_path: str = "name_gender_stats.csv", debug: bool = True):
        """
        Initialize the GenderGuesser with the name statistics data.
        
        Args:
            data_file_path (str): Path to the CSV file containing name gender statistics.
                                 Default is "name_gender_stats.csv" in the current directory.
            debug (bool): If True, provides detailed debugging information
        """
        self.data_file_path = data_file_path
        self.gender_data = None
        self.debug = debug
        self.name_lookup = {}
        
        if self.debug:
            print("🔍 DEBUG: Initializing GenderGuesser...")
            print(f"🔍 DEBUG: Looking for data file: {self.data_file_path}")
            print(f"🔍 DEBUG: Current working directory: {os.getcwd()}")
            
        self._load_gender_data()
    
    def _load_gender_data(self):
        """
        Private method to load the gender statistics data into memory.
        This creates a lookup dictionary for fast name-to-gender mapping.
        """
        try:
            # Check if file exists first
            if not os.path.exists(self.data_file_path):
                available_files = [f for f in os.listdir('.') if f.endswith('.csv')]
                error_msg = f"""
❌ ERROR: Cannot find the data file '{self.data_file_path}'

DEBUGGING INFORMATION:
- Current directory: {os.getcwd()}
- Available CSV files in current directory: {available_files if available_files else 'None found'}

SOLUTIONS:
1. Make sure 'name_gender_stats.csv' is in the same directory as this script
2. Check the file name spelling (case sensitive)
3. If the file is elsewhere, provide the full path:
   GenderGuesser('/path/to/your/name_gender_stats.csv')

EXAMPLE:
   guesser = GenderGuesser('C:/Users/YourName/Documents/name_gender_stats.csv')
"""
                raise FileNotFoundError(error_msg)
            
            # Load the CSV file with name statistics
            if self.debug:
                print(f"✅ DEBUG: Found data file, loading gender statistics...")
                
            self.gender_data = pd.read_csv(self.data_file_path)
            
            # Validate the data format
            required_columns = ['Name', 'Percent_Male']
            missing_columns = [col for col in required_columns if col not in self.gender_data.columns]
            
            if missing_columns:
                error_msg = f"""
❌ ERROR: Invalid data file format

REQUIRED COLUMNS: {required_columns}
FOUND COLUMNS: {list(self.gender_data.columns)}
MISSING COLUMNS: {missing_columns}

SOLUTION:
Make sure your CSV file has the correct format with columns:
- 'Name': containing the first names
- 'Percent_Male': containing the percentage (0-100) of male usage
"""
                raise ValueError(error_msg)
            
            # Check for empty data
            if len(self.gender_data) == 0:
                raise ValueError("❌ ERROR: The data file is empty. Please check your name_gender_stats.csv file.")
            
            # Convert to dictionary for faster lookups
            # Key: name (lowercase), Value: Percent_Male
            self.name_lookup = dict(zip(
                self.gender_data['Name'].str.lower(), 
                self.gender_data['Percent_Male']
            ))
            
            # Data quality check
            null_names = self.gender_data['Name'].isnull().sum()
            null_percentages = self.gender_data['Percent_Male'].isnull().sum()
            
            if self.debug:
                print(f"✅ DEBUG: Successfully loaded {len(self.name_lookup):,} names from the database")
                print(f"🔍 DEBUG: Data quality check:")
                print(f"   - Total names: {len(self.gender_data):,}")
                print(f"   - Null names: {null_names}")
                print(f"   - Null percentages: {null_percentages}")
                print(f"   - Sample names: {list(self.gender_data['Name'].head().values)}")
                print(f"   - Percentage range: {self.gender_data['Percent_Male'].min():.1f}% to {self.gender_data['Percent_Male'].max():.1f}%")
            
            if null_names > 0 or null_percentages > 0:
                warnings.warn(f"⚠️ WARNING: Found {null_names} null names and {null_percentages} null percentages in data")
            
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            error_msg = f"""
❌ ERROR: Problem loading gender data from '{self.data_file_path}'

ORIGINAL ERROR: {str(e)}

DEBUGGING STEPS:
1. Check if the file exists and is readable
2. Verify the file is a valid CSV format
3. Make sure the file isn't corrupted or empty
4. Try opening the file in a text editor to check its contents

PYTHON COMMAND TO TEST FILE:
import pandas as pd
df = pd.read_csv('{self.data_file_path}')
print(df.head())
print(df.columns)
"""
            raise Exception(error_msg)
    
    def predict_gender(self, name: str) -> dict:
        """
        Predict gender for a single name.
        
        Args:
            name (str): The first name to analyze
            
        Returns:
            dict: Contains 'percent_male', 'gender_code', and 'found_in_database'
        """
        # Input validation and debugging
        if pd.isnull(name) or name is None:
            if self.debug:
                print(f"🔍 DEBUG: Received null/None name, returning Unknown")
            return {
                'percent_male': None,
                'gender_code': "U",
                'found_in_database': False
            }
        
        # Convert name to lowercase for matching
        original_name = str(name)
        name_lower = original_name.lower().strip()
        
        if self.debug:
            print(f"🔍 DEBUG: Processing name '{original_name}' -> '{name_lower}'")
        
        # Check for empty or invalid names
        if not name_lower:
            if self.debug:
                print(f"🔍 DEBUG: Empty name after processing, returning Unknown")
            return {
                'percent_male': None,
                'gender_code': "U",
                'found_in_database': False
            }
        
        # Look up the name in our database
        if name_lower in self.name_lookup:
            percent_male = self.name_lookup[name_lower]
            
            # Apply gender classification rules
            if percent_male > 55:
                gender_code = "M"  # Male
            elif percent_male < 45:
                gender_code = "F"  # Female
            else:  # 45 <= percent_male <= 55
                gender_code = "A"  # Ambiguous
            
            if self.debug:
                print(f"✅ DEBUG: Found '{name_lower}' - {percent_male:.1f}% male -> {gender_code}")
                
            return {
                'percent_male': percent_male,
                'gender_code': gender_code,
                'found_in_database': True
            }
        else:
            # Name not found in database - provide helpful suggestions
            if self.debug:
                # Look for similar names
                similar_names = [n for n in self.name_lookup.keys() if n.startswith(name_lower[:2]) and len(n) >= 2][:5]
                print(f"❌ DEBUG: Name '{name_lower}' not found in database")
                if similar_names:
                    print(f"🔍 DEBUG: Similar names in database: {similar_names}")
            
            return {
                'percent_male': None,
                'gender_code': "U",  # Unknown
                'found_in_database': False
            }
    
    def apply_gender_to_dataframe(self, 
                                 df: pd.DataFrame, 
                                 name_column: str,
                                 output_file_path: Optional[str] = None,
                                 include_stats: bool = True) -> pd.DataFrame:
        """
        Apply gender prediction to an entire DataFrame.
        
        This is the main function data analysts will use to add gender predictions
        to their datasets.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing names to analyze
            name_column (str): The name of the column containing first names
            output_file_path (str, optional): If provided, saves the result to this file.
                                            Recommended: Use .csv for large datasets
                                            .xlsx requires openpyxl and has performance/size limits
            include_stats (bool): If True, includes additional statistical columns
            
        Returns:
            pd.DataFrame: Original DataFrame with new gender prediction columns added
        """
        
        # Comprehensive input validation with detailed debugging
        if self.debug:
            print(f"\n🔍 DEBUG: Starting DataFrame validation...")
            print(f"🔍 DEBUG: DataFrame type: {type(df)}")
            if hasattr(df, 'shape'):
                print(f"🔍 DEBUG: DataFrame shape: {df.shape}")
        
        if not isinstance(df, pd.DataFrame):
            error_msg = f"""
❌ ERROR: Input must be a pandas DataFrame

RECEIVED TYPE: {type(df)}

SOLUTION:
Make sure you're passing a pandas DataFrame. Example:
import pandas as pd
df = pd.read_csv('your_file.csv')  # This creates a DataFrame
result = guesser.apply_gender_to_dataframe(df, 'name_column')
"""
            raise ValueError(error_msg)
        
        if df.empty:
            error_msg = f"""
❌ ERROR: DataFrame is empty (no rows)

SOLUTIONS:
1. Check if your data file loaded correctly
2. Verify your data isn't filtered to zero rows
3. Try: print(len(df)) to check row count
4. Try: print(df.head()) to see your data
"""
            raise ValueError(error_msg)
        
        # Detailed column validation
        if self.debug:
            print(f"🔍 DEBUG: Available columns: {list(df.columns)}")
            print(f"🔍 DEBUG: Looking for column: '{name_column}'")
        
        if name_column not in df.columns:
            # Provide helpful suggestions for column names
            similar_columns = [col for col in df.columns if 'name' in col.lower()]
            error_msg = f"""
❌ ERROR: Column '{name_column}' not found in DataFrame

AVAILABLE COLUMNS: {list(df.columns)}
COLUMNS WITH 'name': {similar_columns if similar_columns else 'None found'}

SOLUTIONS:
1. Check the spelling of your column name (case sensitive)
2. Use one of the available column names listed above
3. Print your column names: print(df.columns.tolist())

EXAMPLE:
If your column is actually 'Name' or 'first_name', use:
result = guesser.apply_gender_to_dataframe(df, 'Name')
"""
            raise ValueError(error_msg)
        
        # Check for null values in the name column
        null_count = df[name_column].isnull().sum()
        empty_count = (df[name_column] == '').sum() if df[name_column].dtype == 'object' else 0
        
        if self.debug:
            print(f"🔍 DEBUG: Data quality in '{name_column}' column:")
            print(f"   - Total rows: {len(df):,}")
            print(f"   - Null values: {null_count:,}")
            print(f"   - Empty strings: {empty_count:,}")
            print(f"   - Valid names: {len(df) - null_count - empty_count:,}")
            if len(df) > 0:
                sample_names = df[name_column].dropna().head(5).tolist()
                print(f"   - Sample names: {sample_names}")
        
        if null_count > 0 or empty_count > 0:
            total_invalid = null_count + empty_count
            percentage = (total_invalid / len(df)) * 100
            if percentage > 50:
                warning_msg = f"""
⚠️ WARNING: {total_invalid:,} out of {len(df):,} names ({percentage:.1f}%) are null or empty
This will result in many 'U' (Unknown) classifications.

SUGGESTIONS:
1. Check your data loading process
2. Verify the correct column name
3. Clean your data before processing
"""
                print(warning_msg)
        
        # Warn about Excel format for large datasets
        if output_file_path and output_file_path.lower().endswith('.xlsx'):
            if len(df) > 100000:  # Warn for datasets > 100k rows
                warnings.warn(
                    f"⚠️ WARNING: You're saving {len(df):,} rows to Excel format. "
                    "Consider using .csv for better performance and compatibility. "
                    "Excel has a ~1 million row limit and is much slower than CSV.",
                    UserWarning
                )
        
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        print(f"\n📊 PROCESSING STARTED")
        print(f"Analyzing {len(df):,} records...")
        print(f"Looking up names in column: '{name_column}'")
        
        if self.debug:
            print(f"🔍 DEBUG: Starting gender prediction for {len(df)} names...")
        
        # Apply gender prediction to each name
        gender_predictions = result_df[name_column].apply(self.predict_gender)
        
        # Extract results into separate columns
        result_df['Percent_Male'] = gender_predictions.apply(lambda x: x['percent_male'])
        result_df['Gender_Code'] = gender_predictions.apply(lambda x: x['gender_code'])
        
        if include_stats:
            result_df['Found_In_Database'] = gender_predictions.apply(lambda x: x['found_in_database'])
        
        if self.debug:
            print(f"✅ DEBUG: Completed gender prediction")
            # Show distribution of results
            code_counts = result_df['Gender_Code'].value_counts()
            print(f"🔍 DEBUG: Results distribution: {dict(code_counts)}")
        
        # Generate summary statistics
        self._print_summary_stats(result_df, name_column)
        
        # Save to file if requested
        if output_file_path:
            self._save_results(result_df, output_file_path)
        
        return result_df
    
    def _print_summary_stats(self, df: pd.DataFrame, name_column: str):
        """
        Print detailed summary statistics about the gender prediction results.
        """
        total_records = len(df)
        found_count = df['Gender_Code'].ne('U').sum()
        not_found_count = df['Gender_Code'].eq('U').sum()
        
        male_count = df['Gender_Code'].eq('M').sum()
        female_count = df['Gender_Code'].eq('F').sum()
        ambiguous_count = df['Gender_Code'].eq('A').sum()
        
        print("\n" + "="*70)
        print("📊 GENDER PREDICTION SUMMARY")
        print("="*70)
        print(f"Total records processed: {total_records:,}")
        print(f"Names found in database: {found_count:,} ({found_count/total_records*100:.1f}%)")
        print(f"Names not found: {not_found_count:,} ({not_found_count/total_records*100:.1f}%)")
        print()
        print("GENDER DISTRIBUTION:")
        print(f"  👨 Male (M):      {male_count:,} ({male_count/total_records*100:.1f}%)")
        print(f"  👩 Female (F):    {female_count:,} ({female_count/total_records*100:.1f}%)")
        print(f"  ⚖️  Ambiguous (A): {ambiguous_count:,} ({ambiguous_count/total_records*100:.1f}%)")
        print(f"  ❓ Unknown (U):   {not_found_count:,} ({not_found_count/total_records*100:.1f}%)")
        
        # Data quality insights
        if not_found_count > found_count * 0.5:  # More than 50% not found
            print("\n⚠️  DATA QUALITY ALERT:")
            print("   More than 50% of names were not found in the database.")
            print("   This could indicate:")
            print("   - Non-English names (database is US-focused)")
            print("   - Nicknames or unusual name formats")
            print("   - Data quality issues in your input")
        
        print("="*70)
        
        if self.debug:
            # Show some examples of each category
            examples = {}
            for code, label in [('M', 'Male'), ('F', 'Female'), ('A', 'Ambiguous'), ('U', 'Unknown')]:
                code_examples = df[df['Gender_Code'] == code][name_column].head(3).tolist()
                if code_examples:
                    examples[label] = code_examples
            
            if examples:
                print("🔍 DEBUG: Examples by category:")
                for category, names in examples.items():
                    print(f"   {category}: {names}")
                print()
    
    def _save_results(self, df: pd.DataFrame, file_path: str):
        """
        Save the results to a file with comprehensive error handling and debugging.
        """
        if self.debug:
            print(f"🔍 DEBUG: Attempting to save {len(df)} rows to: {file_path}")
        
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if self.debug:
                print(f"🔍 DEBUG: Detected file extension: '{file_extension}'")
                print(f"🔍 DEBUG: File size estimate: ~{len(df) * len(df.columns) * 10} bytes")
            
            if file_extension == '.csv' or not file_extension:
                # Default to CSV - fastest and most compatible
                if not file_extension:
                    file_path += '.csv'
                    if self.debug:
                        print(f"🔍 DEBUG: No extension provided, using CSV: {file_path}")
                
                df.to_csv(file_path, index=False)
                
            elif file_extension == '.xlsx':
                # Excel format - requires openpyxl
                if self.debug:
                    print(f"🔍 DEBUG: Excel format requested, checking openpyxl...")
                try:
                    df.to_excel(file_path, index=False)
                except ImportError as ie:
                    error_msg = f"""
❌ ERROR: Cannot save to Excel format. Missing 'openpyxl' library.

SOLUTION:
Install openpyxl: pip install openpyxl

ALTERNATIVE:
Use CSV format instead (recommended for large datasets):
- Change your file extension from .xlsx to .csv
- CSV files are faster, smaller, and more compatible

FALLBACK:
Saving as CSV instead...
"""
                    print(error_msg)
                    csv_path = file_path.replace('.xlsx', '.csv')
                    df.to_csv(csv_path, index=False)
                    print(f"✅ Saved as CSV: {csv_path}")
                    return
                    
            elif file_extension == '.parquet':
                # Parquet format - requires pyarrow
                if self.debug:
                    print(f"🔍 DEBUG: Parquet format requested, checking pyarrow...")
                try:
                    df.to_parquet(file_path, index=False)
                except ImportError:
                    error_msg = f"""
❌ ERROR: Cannot save to Parquet format. Missing 'pyarrow' library.

SOLUTION:
Install pyarrow: pip install pyarrow

ALTERNATIVE:
Use CSV format instead:
- Change your file extension from .parquet to .csv

FALLBACK:
Saving as CSV instead...
"""
                    print(error_msg)
                    csv_path = file_path.replace('.parquet', '.csv')
                    df.to_csv(csv_path, index=False)
                    print(f"✅ Saved as CSV: {csv_path}")
                    return
            else:
                # Unknown extension - default to CSV
                print(f"⚠️ WARNING: Unknown file extension '{file_extension}'. Using CSV format.")
                file_path = file_path.replace(file_extension, '.csv')
                df.to_csv(file_path, index=False)
            
            # Verify file was created successfully
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✅ Results saved successfully!")
                print(f"   📁 File: {file_path}")
                print(f"   📊 Size: {file_size:,} bytes")
                if self.debug:
                    print(f"🔍 DEBUG: File verification successful")
            else:
                print(f"⚠️ WARNING: File may not have been created successfully")
            
        except PermissionError as pe:
            error_msg = f"""
❌ ERROR: Permission denied when saving to '{file_path}'

POSSIBLE CAUSES:
1. File is open in Excel or another program
2. Insufficient write permissions to the directory
3. File is read-only

SOLUTIONS:
1. Close the file if it's open in another program
2. Try saving to a different location
3. Run Python as administrator (if needed)
4. Check file permissions

ALTERNATIVE:
Try a different filename: {file_path.replace('.', '_new.')}
"""
            print(error_msg)
            
        except Exception as e:
            error_msg = f"""
❌ ERROR: Could not save file to '{file_path}'

ORIGINAL ERROR: {str(e)}

DEBUGGING STEPS:
1. Check if the directory exists: {os.path.dirname(file_path) or 'current directory'}
2. Verify you have write permissions
3. Try a simpler filename without special characters
4. Check available disk space

FALLBACK ATTEMPT:
Trying to save as CSV in current directory...
"""
            print(error_msg)
            
            # Try saving as CSV as fallback
            try:
                fallback_path = f"gender_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(fallback_path, index=False)
                print(f"✅ Saved as CSV fallback: {fallback_path}")
            except Exception as e2:
                print(f"❌ Could not save fallback CSV either: {str(e2)}")
                print("💡 TIP: Try manually specifying a simple filename like 'results.csv'")


# Convenience function for quick usage with enhanced error handling
def predict_gender_for_dataframe(df: pd.DataFrame, 
                                name_column: str,
                                data_file_path: str = "name_gender_stats.csv",
                                output_file_path: Optional[str] = None,
                                include_stats: bool = True,
                                debug: bool = True) -> pd.DataFrame:
    """
    Convenience function to quickly apply gender prediction to a DataFrame.
    
    This function is perfect for analysts who want to use the tool without 
    creating a GenderGuesser object first.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing names to analyze
        name_column (str): The name of the column containing first names
        data_file_path (str): Path to the name gender statistics CSV file
        output_file_path (str, optional): If provided, saves the result to this file
                                        Recommended: Use .csv for large datasets
        include_stats (bool): If True, includes additional statistical columns
        debug (bool): If True, provides detailed debugging information
        
    Returns:
        pd.DataFrame: Original DataFrame with new gender prediction columns added
        
    Example:
        >>> import pandas as pd
        >>> from Gender_Guessing_Tool import predict_gender_for_dataframe
        >>> 
        >>> # Load your data
        >>> my_data = pd.read_csv('customer_data.csv')
        >>> 
        >>> # Apply gender prediction (CSV recommended for large datasets)
        >>> result = predict_gender_for_dataframe(
        ...     df=my_data,
        ...     name_column='customer_first_name',
        ...     output_file_path='customer_data_with_gender.csv'
        ... )
        >>> 
        >>> # Now you have columns: 'Percent_Male', 'Gender_Code', 'Found_In_Database'
    """
    try:
        if debug:
            print("🚀 Starting Gender Prediction using convenience function...")
        
        guesser = GenderGuesser(data_file_path, debug=debug)
        return guesser.apply_gender_to_dataframe(df, name_column, output_file_path, include_stats)
        
    except Exception as e:
        error_msg = f"""
❌ ERROR in predict_gender_for_dataframe()

ORIGINAL ERROR: {str(e)}

COMMON SOLUTIONS:
1. Check that 'name_gender_stats.csv' is in your current directory
2. Verify your DataFrame and column name are correct
3. Try with debug=True for more detailed information

EXAMPLE USAGE:
result = predict_gender_for_dataframe(
    df=your_dataframe,
    name_column='your_name_column',  # Check this column exists
    debug=True  # For detailed debugging
)
"""
        print(error_msg)
        raise


# Example usage and demonstration
if __name__ == "__main__":
    """
    Example usage of the Gender Guessing Tool.
    
    This section demonstrates how to use the package with sample data.
    """
    print("🎯 Gender Guessing Tool - Example Usage")
    print("="*50)
    
    # Example 1: Create sample data
    sample_data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'first_name': ['James', 'Maria', 'Alex', 'Jordan', 'Michael', 'Jessica', 'Chris', 'UnknownName'],
        'last_name': ['Smith', 'Garcia', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis'],
        'age': [25, 30, 22, 35, 28, 26, 31, 29]
    })
    
    print("📋 Sample input data:")
    print(sample_data)
    print()
    
    try:
        # Example 2: Using the convenience function (CSV recommended)
        print("🔧 Method 1: Using convenience function")
        result_df = predict_gender_for_dataframe(
            df=sample_data,
            name_column='first_name',
            output_file_path='sample_results.csv',  # CSV for best performance
            debug=True
        )
        
        print("\n📊 Result with gender predictions:")
        print(result_df[['first_name', 'Percent_Male', 'Gender_Code']])
        
    except Exception as e:
        print(f"❌ Error running example: {str(e)}")
        print("\n💡 TROUBLESHOOTING:")
        print("1. Make sure 'name_gender_stats.csv' is in the same directory as this script")
        print("2. Check that pandas is installed: pip install pandas")
        print("3. Verify file permissions and available disk space")
        
        # Show how to use with class-based approach
        print("\n🔧 Method 2: Using GenderGuesser class")
        print("guesser = GenderGuesser('path/to/name_gender_stats.csv')")
        print("result = guesser.apply_gender_to_dataframe(df, 'first_name')")


"""
================================================================================
DETAILED USAGE INSTRUCTIONS FOR DATA ANALYSTS
================================================================================

📚 1. QUICK START:
   - Place this file and 'name_gender_stats.csv' in your project directory
   - Import: from Gender_Guessing_Tool import predict_gender_for_dataframe
   - Use: result = predict_gender_for_dataframe(your_df, 'name_column')

📊 2. OUTPUT COLUMNS:
   - 'Percent_Male': Percentage likelihood the name is male (0-100)
   - 'Gender_Code': 
     * "M" = Male (>55% male)
     * "F" = Female (<45% male) 
     * "A" = Ambiguous (45-55% male)
     * "U" = Unknown (name not in database)
   - 'Found_In_Database': Boolean indicating if name was found

💾 3. FILE FORMAT RECOMMENDATIONS:
   - CSV (.csv): ✅ RECOMMENDED - Fast, unlimited size, universally compatible
   - Excel (.xlsx): ⚠️ Requires 'pip install openpyxl', ~1M row limit, slower
   - Parquet (.parquet): 📦 Requires 'pip install pyarrow', good for very large datasets

⚡ 4. PERFORMANCE:
   - The tool loads all name data into memory for fast lookups
   - Suitable for datasets up to millions of records
   - Processing time is roughly linear with dataset size
   - CSV output is fastest for large datasets

📋 5. DATA REQUIREMENTS:
   - Input: Any pandas DataFrame with a column containing first names
   - Names are automatically converted to lowercase for matching
   - Works with any file format pandas can read (CSV, Excel, Parquet, etc.)

🐛 6. DEBUGGING:
   - Set debug=True for detailed information about what the tool is doing
   - Check error messages carefully - they include specific solutions
   - Use the step-by-step guide at the top of this file

❗ 7. COMMON ISSUES & SOLUTIONS:
   - "Module not found": Place the .py file in your current directory
   - "File not found": Ensure name_gender_stats.csv is in the same folder
   - "Column not found": Check column name spelling (case sensitive)
   - High "Unknown" rate: Normal for non-English names or unusual spellings
   - Excel errors: Install openpyxl or use CSV format
   - Permission errors: Close files if open in other programs

📞 8. SUPPORT:
   - Use debug=True for detailed troubleshooting information
   - Check the step-by-step guide at the top of this file
   - Error messages include specific solutions for most issues
================================================================================
"""