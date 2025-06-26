from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy import stats
import math
from scipy.stats import norm, t, chi2, f
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Python Hypothesis Testing Service is running"})

@app.route('/hypothesis-test', methods=['POST'])
def hypothesis_test():
    try:
        data = request.json
        
        num_samples = data['numSamples']
        claim_type = data['claimType']
        p_value = data['pValue']
        csv_data = data['csvData']
        is_known = data['isKnown']
        hypothesized_value = data['hypothesizedValue']
        known_param = data.get('knownParam')
        known_param2 = data.get('knownParam2')
        claim_direction = data['claimDirection']
        
        # Parse CSV data
        df = pd.read_csv(io.StringIO(csv_data), header=None)
        
        if num_samples == 1:
            data_sample = df[0].dropna().tolist()
        else:
            data1 = df[0].dropna().tolist()
            data2 = df[1].dropna().tolist()
        
        result = {"conclusion": "", "details": {}}
        
        if claim_type == 1:  # Mean testing
            mu0 = hypothesized_value
            
            if num_samples == 1:
                sample = data_sample
                n = len(sample)
                sample_mean = np.mean(sample)
                result["details"]["sample_size"] = n
                result["details"]["sample_mean"] = sample_mean
                
                if is_known == "yes":
                    sigma = known_param
                    if claim_direction == "greater":
                        if sample_mean >= mu0 + (sigma / math.sqrt(n)) * norm.ppf(1 - p_value):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "less":
                        if sample_mean <= mu0 - (sigma / math.sqrt(n)) * norm.ppf(1 - p_value):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "equal":
                        if abs(sample_mean - mu0) >= (sigma / math.sqrt(n)) * norm.ppf(1 - p_value / 2):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                else:
                    sample_variance = sum((x - sample_mean) ** 2 for x in sample) / (n - 1)
                    sample_std_dev = math.sqrt(sample_variance)
                    result["details"]["sample_std"] = sample_std_dev
                    
                    if claim_direction == "greater":
                        t_critical = t.ppf(1 - p_value, n - 1)
                        if sample_mean >= mu0 + (sample_std_dev / math.sqrt(n)) * t_critical:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "less":
                        t_critical = t.ppf(1 - p_value, n - 1)
                        if sample_mean <= mu0 - (sample_std_dev / math.sqrt(n)) * t_critical:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "equal":
                        if abs(sample_mean - mu0) >= (sample_std_dev / math.sqrt(n)) * t.ppf(1 - p_value / 2, n - 1):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
            
            else:  # Two samples
                n = len(data1)
                m = len(data2)
                sample_mean1 = np.mean(data1)
                sample_mean2 = np.mean(data2)
                result["details"]["sample_size1"] = n
                result["details"]["sample_size2"] = m
                result["details"]["sample_mean1"] = sample_mean1
                result["details"]["sample_mean2"] = sample_mean2
                
                if is_known == "yes":
                    sigma1 = known_param
                    sigma2 = known_param2
                    if claim_direction == "greater":
                        z_critical = norm.ppf(1 - p_value)
                        if (sample_mean1 - sample_mean2) >= mu0 + math.sqrt((sigma1**2 / n) + (sigma2**2 / m)) * z_critical:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "less":
                        z_critical = norm.ppf(1 - p_value)
                        if (sample_mean1 - sample_mean2) <= mu0 - math.sqrt((sigma1**2 / n) + (sigma2**2 / m)) * z_critical:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "equal":
                        z_critical = norm.ppf(1 - p_value / 2)
                        if abs(sample_mean1 - sample_mean2 - mu0) >= math.sqrt((sigma1**2 / n) + (sigma2**2 / m)) * z_critical:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                else:
                    sample_variance1 = sum((x - sample_mean1) ** 2 for x in data1) / (n - 1)
                    sample_variance2 = sum((x - sample_mean2) ** 2 for x in data2) / (m - 1)
                    sp = math.sqrt(((n - 1) * sample_variance1 + (m - 1) * sample_variance2) / (n + m - 2))
                    result["details"]["sample_variance1"] = sample_variance1
                    result["details"]["sample_variance2"] = sample_variance2
                    
                    if claim_direction == "greater":
                        if (sample_mean1 - sample_mean2) >= mu0 + sp * math.sqrt(1/n + 1/m) * t.ppf(1 - p_value, n + m - 2):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "less":
                        if (sample_mean1 - sample_mean2) <= mu0 - sp * math.sqrt(1/n + 1/m) * t.ppf(1 - p_value, n + m - 2):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "equal":
                        if abs(sample_mean1 - sample_mean2 - mu0) >= sp * math.sqrt(1/n + 1/m) * t.ppf(1 - p_value / 2, n + m - 2):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
        
        elif claim_type == 2:  # Standard deviation testing
            if num_samples == 1:
                sigma0 = hypothesized_value
                n = len(data_sample)
                result["details"]["sample_size"] = n
                
                if is_known == "yes":
                    mu = known_param
                    sample_variance = sum((x - mu) ** 2 for x in data_sample)
                    if claim_direction == "greater":
                        chi_critical = chi2.ppf(1 - p_value, n)
                        if sample_variance >= sigma0**2 * chi_critical:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "less":
                        chi_critical = chi2.ppf(p_value, n)
                        if sample_variance <= sigma0**2 * chi_critical:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "equal":
                        if sample_variance <= sigma0**2 * chi2.ppf(1 - p_value / 2, n) or sample_variance >= sigma0**2 * chi2.ppf(p_value / 2, n):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                else:
                    sample_mean = np.mean(data_sample)
                    sample_variance = sum((x - sample_mean) ** 2 for x in data_sample) / (n - 1)
                    result["details"]["sample_mean"] = sample_mean
                    result["details"]["sample_variance1"] = sample_variance
                    
                    if claim_direction == "greater":
                        chi_critical = chi2.ppf(1 - p_value, n - 1)
                        if sample_variance >= sigma0**2 * chi_critical / (n - 1):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "less":
                        chi_critical = chi2.ppf(p_value, n - 1)
                        if sample_variance <= sigma0**2 * chi_critical / (n - 1):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "equal":
                        if sample_variance <= sigma0**2 * chi2.ppf(1 - p_value / 2, n - 1) / (n - 1) or sample_variance >= sigma0**2 * chi2.ppf(p_value / 2, n - 1) / (n - 1):
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
            
            else:  # Two samples
                n = len(data1)
                m = len(data2)
                result["details"]["sample_size1"] = n
                result["details"]["sample_size2"] = m
                
                if is_known == "yes":
                    mu1 = known_param
                    mu2 = known_param2
                    sample_variance1 = sum((x - mu1) ** 2 for x in data1)
                    sample_variance2 = sum((x - mu2) ** 2 for x in data2)
                    result["details"]["sample_variance1"] = sample_variance1
                    result["details"]["sample_variance2"] = sample_variance2
                    
                    if claim_direction == "greater":
                        f_critical = f.ppf(1 - p_value, m, n)
                        if (sample_variance1 / sample_variance2) >= f_critical * m / n:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "less":
                        f_critical = f.ppf(1 - p_value, n, m)
                        if (sample_variance1 / sample_variance2) <= f_critical * n / m:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "equal":
                        f_critical1 = f.ppf(1 - p_value / 2, m, n)
                        f_critical2 = f.ppf(p_value / 2, m, n)
                        if (sample_variance1 / sample_variance2) >= f_critical1 * m / n or (sample_variance1 / sample_variance2) <= f_critical2 * m / n:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                else:
                    sample_variance1 = sum((x - np.mean(data1)) ** 2 for x in data1) / (n - 1)
                    sample_variance2 = sum((x - np.mean(data2)) ** 2 for x in data2) / (m - 1)
                    result["details"]["sample_variance1"] = sample_variance1
                    result["details"]["sample_variance2"] = sample_variance2
                    
                    if claim_direction == "greater":
                        f_critical = f.ppf(1 - p_value, n - 1, m - 1)
                        if (sample_variance1 / sample_variance2) >= f_critical:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "less":
                        f_critical = f.ppf(1 - p_value, m - 1, n - 1)
                        if (sample_variance2 / sample_variance1) >= f_critical:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
                    elif claim_direction == "equal":
                        f_critical1 = f.ppf(1 - p_value / 2, n - 1, m - 1)
                        f_critical2 = f.ppf(p_value / 2, n - 1, m - 1)
                        if (sample_variance1 / sample_variance2) >= f_critical1 or (sample_variance2 / sample_variance1) <= f_critical2:
                            result["conclusion"] = "Reject the null hypothesis."
                        else:
                            result["conclusion"] = "Fail to reject the null hypothesis."
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
