"""

Created on Wed May 24 23:47:21 2023

@author: andrew7

Univariate Forecasting Framework
"""
import itertools
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler
)
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error
)


# Statistical software libraries;
from statsmodels.tsa.stattools import adfuller, kpss, breakvar_heteroskedasticity_test
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import ExponentialSmoothing

# Filter warnings;
warnings.filterwarnings("ignore")

# Floating point number display formatting;
pd.set_option('display.float_format', lambda x: '%.5f' % x)  # pandas
np.set_printoptions(precision=5, suppress=True)  # numpy

# DataFrame display formatting;
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# Log file formatting;
LOG_FMT = """
        %(levelname)s: %(funcName)s: %(name)s: %(asctime)s:
        %(message)s (Line: %(lineno)s [%(filename)s])
        """
YMD_FMT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = "model_process.log"

# logging protocol configuration;
# logging.basicConfig(
# format=log_fmt,
# filename=log_file,
# datefmt=ymd_fmt,
# level=# logging.INFO)


class SimpleProcess():
    """
     -----------------------------------------
    ===========================================

             Simple Process:

                Models:

        * Exp. Smoothing

        * Auto-Regressive(p)

    ===========================================
     -----------------------------------------
    """

    def __init__(self, directory: str, y_name="Streams", window=45):
        """
         -----------------------------------------
        ===========================================

         Initializes SimpleProcess class instance;

          params:
          * directory: (str), Directory to iterate over ;
          * y_name: (str), Variable to generate forecasts of;
          * window: (int), Length of forecasting horizon;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Directory;
        self.directory = directory

        # Forecasting window;
        self.window = window

        # Endog. Variable of Interest:
        self.y_name = y_name

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Directory: %s ;", directory)
        # logging.info("Forecasting window: %s Days", window)
        # logging.info("Predict: Daily %s", y_name)
        # logging.info("Runtime: %s ;", runtime)

    def locate_file(self, directory: str,):
        """
         -----------------------------------------
        ===========================================

         Iterates over 'directory' to locate modeling dataset;

          params:
          * directory: (str), Directory to iterate over ;
          * y_name: (str), Variable to generate forecasts of;

        ===========================================
         -----------------------------------------
        """

        # pathlib.Path(), iter directory;
        paths = Path(directory).iterdir()

        # Start timing;
        # start = time.perf_counter()

        # For each element in path;
        for path in paths:
            # Looking for files, ending with '.csv';
            if path.is_file() and path.suffix == ".csv":
                check_model = str(path.name).startswith("EST_")
                # Modeling files all start with 'EST'
                if check_model:
                    # logging.info("Modeling dataset found: %s ;", p.name)
                    filename = str(path.name)

                    # File location object; Modeling Data;
                    file_location = "/".join([directory, filename])
                    # logging.info("File location: %s ;", self.file_location)
                else:
                    # logging.info("No modeling dataset found ; ", p.name)
                    print("Dataset not found; Consult documentation")
                    return None

        return file_location

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # # logging;
        # logging.info("Runtime: %s s", runtime)

    def read_file(self, file_location: str, y_name: str):
        """
         -----------------------------------------
        ===========================================

         Processes modeling dataset;

          params:
          * file_location: (str), Directort location of modeling dataset;;
          * y_name: (str), Variable to generate forecasts of;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Read in file from location in Multivariate.path_iterate();
        read_file = pd.read_csv(
            "".join(file_location), sep=",", parse_dates=[0], index_col=[0]
        )

        # Instantiating file object;
        file = read_file.filter(like=y_name)

        # File headers;
        header = list(file.columns)

        # Data type == 'float64' Ex. 1.0, 2.0 ...;
        dtypes_dict = {s: "float64" for s in header}

        # Setting data type;
        for key, value in dtypes_dict.items():
            file[key] = file[key].astype(value)

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        return file

        # logging;
        # logging.info("Runtime: %s s", runtime)

    def generate_y_data(self, file) -> dict:
        """
         -----------------------------------------
        ===========================================

        Isolates initial modeling series data by 'Source':
         ie., Apple Music, SoundCloud, Spotify;

        params:
         * file: (DataFrane) DataFrame containing historical daily level of
          streams data, isolated by Digital Streaming Platform;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Sources of streams;
        sources = list(file.columns)

        # Isolated data series for each source of streams;
        y_data = {source: file[source] for source in sources}

        # For each source of streams;
        for source in sources:
            assert np.isfinite(y_data[source]).all()

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s s", runtime)

        return y_data

    def stationarity(self, data: pd.Series) -> dict:
        """name, series
         -----------------------------------------
        ===========================================

         Returns DataFrame containing test statistics and
          p-values for Augmented Dickey-Fuller, Kwiatkowski–
          Phillips–Schmidt–Shin tests for stationarity;

          params:
          * data: (dict) Dictionary of isolated series of
          historical daily level of streams data;
          * data should == self.variables

          returns:
          * self.[t]rain_[t]est_[s]plits

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Alpha levels for significance;
        alpha_p = 0.05
        alpha_t = 1.96

        # Augmented Dickey_Fuller(ADF) Test Statistic and P-Value;
        a_t, a_p = adfuller(data)[:2]

        # Kwiatkowski-Philips-Schmidt-Shin(KPSS) Test Statistic and P-Value;
        k_t, k_p = kpss(data)[:2]

        # T-Value, P-Value checks;
        t_check = (a_t >= abs(alpha_t)) | (k_t >= abs(alpha_t))
        p_check = (a_p >= alpha_p) | (k_p >= alpha_p)

        # Stationarity criteria;
        stationarity_check = t_check | p_check

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s s", runtime)

        return {
            "Stationarity likely:": stationarity_check,
            "Augmented Dickey-Fuller": {
                "t_stat": round(a_t, 5),
                "p_value": round(a_p, 5)},
            "Kwiatkowksi-Phillips-Schmidt-Shin": {
                "t_stat": round(k_t, 5),
                "p_value": round(k_p, 5)}
        }

    def set_parameters(self,
                       scaler_keys=["robust", "power", "standard", "minmax"],
                       lag_periods=[1, 2, 3, 4, 5, 6, 7],
                       seasonal_periods=[30, 60, 90],
                       seasonal_terms=[1],
                       trends=["c"],
                       simple_iter=True) -> dict:
        """
         -----------------------------------------
        ===========================================

         Sets iteration lists of Baseline and AR(p) model iteration regression
         parameters;

          params:
          * models: (dict), Dictionary indicating models to run, ie.,
          baseline="exp", model="arp"
          * lag_periods: (list), List containing lag orders to iterate;
          * cov_types: (list), List containing covariance types to iterate;
          * scaler_keys (list), List containing scaler transformation to apply;
          * seasonal_periods: (list), List containing seasonal lag_periods to iterate;
          * seasonal_terms: (list), List containing strings indicating inclusion of
          seasonal term;
          * trends: (list), List containing trend terms to include;
          * trend_terms: (list), List containing strings indicating inclusion of
          trend terms;
          * simple_iter: (bool)
              * If True: Estimate one model per specified lag order;
              - Ex. if lag_periods == 3, estimate AR(3);
              * Else: Estimate one model for each step leading up to specified lag order;
              - Ex. if lag_periods == 3, estimate AR(1), ... , AR(3);


          returns:
          * Baseline iteration parameters,
          * Baseline iteration names,
          * Model iteration parameters,
          * Model iteration names;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Lag periods as a range;
        if not simple_iter and len(lag_periods) <= 1:
            lag_periods = list(np.linspace(1, lag_periods[-1]+1,
                                           num=int(lag_periods[-1]+1)).astype(int))
        else:
            pass

        product_keys = ["lag_order", "scaler", "seasonal_order", "seasonal_term",
                        "trend_term"]

        # Unique combinations of parameters;
        product = sorted(
            set(
                list(
                    itertools.product(
                        lag_periods, scaler_keys, seasonal_periods,
                        seasonal_terms, trends))))

        # Model iteration parameters;
        initial_params = {
            "".join(["iter_", str(i)]):
                dict(zip(product_keys, param))
                for i, param in enumerate(product)
        }
        # Model iteration names;
        model_iterations = list(initial_params)

        # Parameter keys;
        # Mapping AR(p) parameters to exp. smoothing parameters;
        ex_map = {
            1: "additive",
            0: None,
            "c": "additive",
            "t": "additive",
            "n": None
        }

        # Baseline model parameters;
        ex_s_params = {
            iteration: {
                "scaler":
                    initial_params[iteration]["scaler"],
                "seasonal_order":
                    initial_params[iteration]["seasonal_order"],
                "trend_term":
                    ex_map[initial_params[iteration]["trend_term"]],
                "seasonal_term":
                    ex_map[initial_params[iteration]["seasonal_term"]]
            } for iteration in model_iterations}

        # Regression parameters;
        model_params = {
            iteration: {
                "ar_p": initial_params[iteration],
                "ex_s": ex_s_params[iteration]
            } for iteration in model_iterations
        }

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start-finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return model_params

    def apply_splits(self,
                     window: int,
                     data,
                     freq="d") -> tuple:
        """
         -----------------------------------------
        ===========================================

         Performs train-test splitting;

          params:
          * window: (int), Length of forecasting horizon;
          * data: (dict), Dictionary of isolated modeling data series;
          * freq: (int), Data frequency;

          returns:
          * Dictionary of train/test split indices/data;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Check frequency of timeseries;
        if freq == "h":
            window = window * 24

        # Performing Train/Test splits on series indices and values;
        data_iv = {
            "train": {
                "index": data[:-window].index,
                "values": data[:-window].values.reshape(-1, 1)},
            "test": {
                "index": data[-window:].index,
                "values": data[-window:].values.reshape(-1, 1)}
        }
        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return data_iv

    def scaler_assign(self,
                      model_params) -> dict:
        """
         -----------------------------------------
        ===========================================

         Assigns pre-processing scale transformations of
          endogenous variable specified in SimpleProcess.set_ar_parameters;

          params:
          * data: (dict), Dictionary containing train/test split data;
          * indices: (dict), Dictionary containing train/test split indexes;

          returns:
          * Dictionary of assigned sklearn scaler instances;

        ===========================================
         -----------------------------------------
        """
        # Start timing;
        # start = time.perf_counter()

        # Container for assigned scaler functions;
        assign_scalers = {}

        # Dictionary of scalers; Baseline + Primary iterations;
        ex_scalers = {
            iteration: model_params[iteration]["ex_s"]["scaler"]
            for iteration in model_params
        }

        # Dictionary of scalers; Baseline + Primary iterations;
        ar_scalers = {
            iteration: model_params[iteration]["ar_p"]["scaler"]
            for iteration in model_params
        }

        # `Asserting` baseline scaler string == primary scaler string;
        match_scalers = (
            all(ex_scalers[iteration] == ar_scalers[iteration]) for iteration in model_params
        )
        # Container;
        assign_scalers = {}

        # Raise Exception if `Assert` is not True;
        if not match_scalers:
            print("Scaler mismatch; Revisit SimpleProcess.set_parameters()")

        for iteration in model_params:
            scaler = np.unique([ex_scalers[iteration], ar_scalers[iteration]][0])
            if scaler == "minmax":
                func = MinMaxScaler
                assign_scalers[iteration] = func()
            if scaler == "power":
                func = PowerTransformer
                assign_scalers[iteration] = func()
            if scaler == "robust":
                func = RobustScaler
                assign_scalers[iteration] = func()
            if scaler == "standard":
                func = StandardScaler
                assign_scalers[iteration] = func()

        assigned_scalers = assign_scalers

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return assigned_scalers

    def scaler_fit(self,
                   data,
                   assigned_scalers,
                   model_params) -> dict:
        """
         -----------------------------------------
        ===========================================

         Fits pre-processing scale transformations of
          endogenous variable specified in SimpleProcess.set_ar_parameters;

          params:
          * data: (dict), Dictionary containing train/test split data;
          * indices: (dict), Dictionary containing train/test split indexes;

          returns:
          * Dictionary of fitted sklearn scaler instances;


        ===========================================
         -----------------------------------------
        """
        # Start timing;
        # start = time.perf_counter()

        # Fitting scalers;
        fit_scalers = {
            iteration: assigned_scalers[iteration].fit(data)
            for iteration in model_params
        }

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return fit_scalers

    def scaler_transform(self,
                         data,
                         fit_scalers,
                         indices,
                         model_params) -> dict:
        """
         -----------------------------------------
        ===========================================

         Applies pre-processing scale transformations of
          endogenous variable specified in SimpleProcess.set_ar_parameters;

          params:
          * data: (dict), Dictionary containing train/test split data;
          * indices: (dict), Dictionary containing train/test split indexes;

          returns:
          * Dictionary of applied sklearn scaler instances;

        ===========================================
         -----------------------------------------
        """
        # Start timing;
        # start = time.perf_counter()

        # Applying scalers;
        transform_scalers = {
            iteration:
                pd.Series(
                    fit_scalers[iteration].transform(data).flatten(),
                    index=indices)
            for iteration in model_params
        }

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return transform_scalers

    def scaler_apply(self,
                     data,
                     indices,
                     model_params) -> dict:
        """
         -----------------------------------------
        ===========================================

         Assigns, fits, applies, and returns output from pre-processing scale transformations
          of endogenous variable;

          params:
          * data: (dict), Dictionary containing train/test split data;
          * indices: (dict), Dictionary containing train/test split indexes;

          returns:
          * Instances of fitted sklearn scaler instances,
          * Dictionary of output from applying sklearn scaler instances;


        ===========================================
         -----------------------------------------
        """
        # Start timing;
        # start = time.perf_counter()

        # SimpleProcess.scaler_assign;
        assigned_scalers = self.scaler_assign(model_params=model_params)

        fit_scalers = self.scaler_fit(
            data=data, assigned_scalers=assigned_scalers, model_params=model_params)

        transformed_data = self.scaler_transform(
            data=data, fit_scalers=fit_scalers, indices=indices, model_params=model_params)

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return (transformed_data, fit_scalers)

    def prepare_regressors(self,
                           data,
                           indices,
                           model_params) -> dict:
        """
         -----------------------------------------
        ===========================================

         Generates initial endog. variable set(s);

          params:
          * data: (dict), Dictionary containing train/test split data;
          * indices: (dict), Dictionary containing train/test split indexes;
          * ex_s_params: (list), List containing Baseline model iteration parameters;

          returns:
          * Instances of fitted sklearn scalers,
          * Dictionary of output from applying sklearn scaler instances;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Endog. series;
        transformed_data, fit_scalers = self.scaler_apply(
            data=data, indices=indices, model_params=model_params)

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return {"transformed_data": transformed_data, "fit_scalers": fit_scalers}
