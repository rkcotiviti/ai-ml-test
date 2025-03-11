import pandas as pd
import numpy as np
import statsmodels.api as sm
from io import BytesIO
import math
import json
import re
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

#new change 2
def convert_to_native(val):
    if isinstance(val, np.int64):
        return int(val)
    elif isinstance(val, np.float64):
        return float(val)
    elif isinstance(val, float) and math.isnan(val):
        return None 
    elif isinstance(val, float) and math.isinf(val):
        return "Infinity" if val > 0 else "NegativeInfinity"  
    return val


def convert_data(data):
    if isinstance(data, dict):
        return {key: convert_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_data(item) for item in data]
    else:
        return convert_to_native(data)


def load_model(input_dir):
    return "dummy"


def create_ols_model(df):
    """ Function to create an OLS model from DataFrame. """
    """ Function to create an OLS model from DataFrame. """
    code = df["MEASURE_CODE"].unique()[0]
    X = df.drop(columns=["CONTRACT_NUMBER", "MEASURE_CODE", "STARYEAR", "STARYEAR_RATE"])
    Y = df["STARYEAR_RATE"]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model, X, Y, code


def extract_model_parameters(model_ols, X,  Y, code):
    try:
        ### Overallfit Table (Adj r2 score) ===> MEASURE_CODE, SOURCE_OF_VARIATION, OVERALL_FIT_VALUE
        model_overallfit = [[code, "AdjustedRSquared", round(model_ols.rsquared_adj, 8)]]
        # model_overallfit_columns = ["MEASURE_CODE", "SOURCE_OF_VARIATION", "OVERALL_FIT_VALUE"]
        # model_overallfit_df = pd.DataFrame(model_overallfit, columns=model_overallfit_columns)

        ### Output Table ===> MEASURE_CODE, INTERCEPT, BACK_YEAR_1_COEFFICIENT, BACK_YEAR_2_COEFFICIENT, BACK_YEAR_3_COEFFICIENT
        model_output = [[code, round(model_ols.params['const'], 8), round(model_ols.params['BACK_YEAR_1_RATE'], 8), 
                         round(model_ols.params['BACK_YEAR_2_RATE'], 8), round(model_ols.params['BACK_YEAR_3_RATE'], 8)]]
        # model_output_columns = ["MEASURE_CODE", "INTERCEPT", "BACK_YEAR_1_COEFFICIENT", "BACK_YEAR_2_COEFFICIENT", 
        #                         "BACK_YEAR_3_COEFFICIENT"]
        # model_output_df = pd.DataFrame(model_output, columns=model_output_columns)

        ### Regression Table ===> MEASURE_CODE, PARAMETER, COEFFICIENT, P_VALUE, STANDARD_ERROR, VIF
        paramaeters = model_ols.params
        p_values = model_ols.pvalues
        std_err = model_ols.bse
        vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        model_regression = [[code, "Intercept", paramaeters[0], p_values[0], std_err[0], np.nan],
                       [code, "BACK_YEAR_3_RATE", paramaeters[1], p_values[1], std_err[1], vifs[0]],
                       [code, "BACK_YEAR_2_RATE", paramaeters[2], p_values[2], std_err[2], vifs[1]],
                       [code, "BACK_YEAR_1_RATE", paramaeters[3], p_values[3], std_err[3], vifs[2]],]
        # model_regression_columns = ["MEASURE_CODE", "PARAMETER", "COEFFICIENT", "P_VALUE", "STANDARD_ERROR", "VIF"]
        # model_regression_df = pd.DataFrame(model_regression, columns=model_regression_columns)

        # Anova Table ===> MEASURE_CODE, SOURCE_OF_VARIATION, DF, P_VALUE
        ss_total = np.sum((Y - np.mean(Y))**2)
        ss_regression = np.sum((model_ols.fittedvalues - np.mean(Y))**2)
        ss_residual = np.sum((Y - model_ols.fittedvalues)**2)

        df_total = len(X) - 1
        df_regression = model_ols.df_model
        df_residual = model_ols.df_resid

        mean_square_regression = ss_regression / df_regression
        mean_square_residual = ss_residual / df_residual
        f_statistic = mean_square_regression / mean_square_residual
        p_value = stats.f.sf(f_statistic, df_regression, df_residual)

        model_anova = [[code, "Regression", df_regression, p_value],
                       [code, "Residuals", df_residual, np.nan],
                       [code, "Total", int(df_regression + df_residual), np.nan]]              
        # model_anova_columns = ["MEASURE_CODE", "SOURCE_OF_VARIATION", "DF", "P_VALUE"]
        # model_anova_df = pd.DataFrame(model_anova, columns=model_anova_columns)
        # model_anova_df['P_VALUE'] = model_anova_df['P_VALUE'].astype(str).replace('nan', None)
        return model_overallfit, model_output, model_regression, model_anova
    
    except Exception as e:
        print("There was a problem with Model Results creation:", e)



def score_unstructured(model, data, query, **kwargs):
    """
    Main function for processing unstructured data, fitting an OLS model, 
    and returning model parameters as a DataFrame.
    """
    try:
        cleaned_data = re.sub(rb'----------------------------[a-zA-Z0-9]+\r\nContent-Disposition: form-data; name="filekey"; filename="[^"]+"\r\nContent-Type: text/csv\r\n\r\n',b'', data)
        cleaned_data = re.sub(rb'\r\n----------------------------[a-zA-Z0-9]+--\r\n', b'', cleaned_data)
        byte_io = BytesIO(cleaned_data)
        df = pd.read_csv(byte_io)
    except Exception as e:
        df = pd.DataFrame()

    model, X, Y, code = create_ols_model(df)
    model_overallfit, model_output, model_regression, model_anova = extract_model_parameters(model, X,  Y, code)
    summary_df = {"model_overallfit_df": model_overallfit, "model_output_df": model_output, 
                  "model_regression_df": model_regression, "model_anova_df": model_anova}
    
    try:
        converted_data = convert_data(summary_df)
        ret = json.dumps(converted_data)
        print("try worked..")
        return ret
    except Exception as e:
        print("exception worked..")
        ret = str(summary_df)
        return ret
