#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:35:12 2023

@author: andrew7
"""
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error
)

# Statistical software libraries;
from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test
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
# logging.basicConfig(format=log_fmt,filename=log_file,datefmt=ymd_fmt,level=# logging.INFO)


class UnivariateBaseline():
    """
     -----------------------------------------
    ===========================================

             Univariate Forecast:

                Models:

        * Exp. Smoothing

        * Auto-Regressive(p)

        * Auto-Regressive (p) Integrated (d) Moving Average (q) () ARIMA

        * Markov Chain Monte Carlo Simulation (MCMC)

        * Long Short-Term Memory Network (LSTM)

    ===========================================
     -----------------------------------------
    """

    def __init__(self,
                 sources,
                 iterations,
                 params,
                 window):
        """
         -----------------------------------------
        ===========================================

         Initializes SimpleProcess class instance;

          params:
          * sources: (list), List containing Digital Streaming Platform (DSP) names;
          * iterations: (list), List containing model iteration names for each DSP;
          * params: (list), List containing regression parameters for each iteration;
          * window: (int), Length of forecasting horizon;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Sources of streams;
        self.sources = sources

        # Model iterations;
        self.iterations = iterations

        # Model iteration parameters;
        self.params = params

        # Forecasting window;
        self.window = window

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Directory: %s ;", directory)
        # logging.info("Forecasting window: %s Days", window)
        # logging.info("Predict: Daily %s", y_name)
        # logging.info("Runtime: %s ;", runtime)

    def run_ex_s(self, transformed_data):
        """
         -----------------------------------------
        ===========================================

         Generates initial ExponentialSmoothing modeling iteration instances;

          params:
          * transformed_data: (dict), Dictionary containing transformed endog. series;

          returns:
          * Instances of fitted sklearn scaler instances,
          * Dictionary of output from applying sklearn scaler instances;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Model parameters;
        params = self.params

        # Model iterations;
        iterations = self.iterations

        # Instances of statsmodels.ExponentialSmoothing;
        ex_s_results = {
            iteration:
                ExponentialSmoothing(
                    endog=transformed_data["train"][iteration],
                    trend=str(params[iteration]["ex_s"]["trend_term"]),
                    seasonal=str(params[iteration]["ex_s"]["seasonal_term"]),
                    seasonal_periods=int(params[iteration]["ex_s"]["seasonal_order"])).fit()
                for iteration in iterations
        }

        # baseline model elements;
        ex_s_instances = {
            iteration: {
                "prediction_instances":  {
                    "i": ex_s_results[iteration].predict(
                        start=0,
                        end=ex_s_results[iteration].model.endog.shape[0] - 1),
                    "o": ex_s_results[iteration].predict(
                        start=ex_s_results[iteration].model.endog.shape[0],
                        end=ex_s_results[iteration].model.endog.shape[0] + self.window - 1)}
            } for iteration in iterations
        }

        return ex_s_instances

    def run_ar_p(self, transformed_data):
        """
         -----------------------------------------
        ===========================================

         Generates initial baseline modeling iteration instances;

          params:
          * transformed_data: (dict), Dictionary containing transformed endog. series;

          returns:
          * Instances of fitted sklearn scaler instances,
          * Dictionary of output from applying sklearn scaler instances;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Model parameters;
        params = self.params

        # Model iterations;
        iterations = self.iterations

        # Instances of statsmodels.ExponentialSmoothing;
        ar_p_results = {
            iteration:
                AutoReg(endog=transformed_data["train"][iteration],
                        lags=int(params[iteration]["ar_p"]["lag_order"]),
                        trend=str(params[iteration]["ar_p"]["trend_term"]),
                        seasonal=bool(params[iteration]["ar_p"]["seasonal_term"]),
                        period=int(params[iteration]["ar_p"]["seasonal_order"])).fit()
            for iteration in iterations
        }

        # baseline model elements;
        ar_p_instances = {
            iteration: {
                "model_instances": ar_p_results[iteration].model,
                "result_instances": ar_p_results[iteration],
                "transformed_data": {
                    "i": transformed_data["train"][iteration],
                    "o": transformed_data["test"][iteration]},
                "prediction_instances":  {
                    "i": ar_p_results[iteration].get_prediction(
                        start=0,
                        end=ar_p_results[iteration].model.endog.shape[0] - 1),
                    "o": ar_p_results[iteration].get_prediction(
                        start=ar_p_results[iteration].model.endog.shape[0],
                        end=ar_p_results[iteration].model.endog.shape[0] + self.window - 1)
                }
            } for iteration in iterations
        }

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return ar_p_instances

    def run_baselines(self,
                      transformed_data):
        """
         -----------------------------------------
        ===========================================

         Generates initial AR(p) modeling iteration instances;

          params:
          * transformed_data: (dict), Dictionary containing transformed endog. series;

          returns:
          * Instances of fitted sklearn scaler instances,
          * Dictionary of output from applying sklearn scaler instances;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Model parameters;
        params = self.params

        # Model iterations;
        iterations = self.iterations

        instances = {
            "ex_s":
                self.run_ex_s(transformed_data=transformed_data),
            "ar_p":
                self.run_ar_p(transformed_data=transformed_data)}

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return instances

    def baseline_predictions(self,
                             source,
                             indices,
                             instances,
                             fit_scalers):
        """
         -----------------------------------------
        ===========================================

         Generates collection of AR(p) modeling iteration instances;

          params:
          * transformed_data: (dict), Dictionary containing transformed endog. series;

          returns:
          * AR(p) model iteration prediction elements;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Model parameters;
        params = self.params

        # Model iterations;
        iterations = self.iterations

        # Max. Lag order == Number of obs. to hold out;
        hold_back = max(params[iteration]["ar_p"]["lag_order"] for iteration in params)

        # Collecting Baseline model iteration predictions;
        initial_predictions = {
            iteration: {
                "i": {
                    "DATE":
                        indices["train"]["index"].values.tolist()[hold_back:],
                    "source":
                        [str(source)]
                    * len(indices["train"]["index"][hold_back:]),
                    "iteration":
                        [str(iteration)]
                    * len(indices["train"]["index"][hold_back:]),
                    "scaler":
                        [str(params[iteration]["ar_p"]["scaler"])]
                    * len(indices["train"]["index"][hold_back:]),
                    "lag_order":
                        [str(params[iteration]["ar_p"]["lag_order"])]
                    * len(indices["train"]["index"][hold_back:]),
                    "seasonal_term":
                        [str(params[iteration]["ar_p"]["seasonal_term"])]
                    * len(indices["train"]["index"][hold_back:]),
                    "seasonal_order":
                        [str(params[iteration]["ar_p"]["seasonal_order"])]
                    * len(indices["train"]["index"][hold_back:]),
                    "trend_term":
                        [str(params[iteration]["ar_p"]["trend_term"])]
                    * len(indices["train"]["index"][hold_back:]),
                    "window_step":
                        [str(-1)]
                        * len(indices["train"]["index"][hold_back:]),
                    "real":
                        fit_scalers["train"][iteration].inverse_transform(
                            instances["ar_p"][iteration]["transformed_data"]["i"].
                            values.reshape(-1, 1)).
                        flatten()[hold_back:],
                    "ex_s_prediction":
                        fit_scalers["train"][iteration].inverse_transform(
                            instances["ex_s"][iteration]["prediction_instances"]["i"].
                            values.reshape(-1, 1)).flatten()[hold_back:],
                    "ar_p_prediction":
                        fit_scalers["train"][iteration].inverse_transform(
                            instances["ar_p"][iteration]["prediction_instances"]["i"].
                            _predicted_mean.reshape(-1, 1))[hold_back:].
                        flatten(),
                    "lower":
                        fit_scalers["train"][iteration].inverse_transform(
                            instances["ar_p"][iteration]["prediction_instances"]["i"].
                            conf_int()["lower"].values.reshape(-1, 1)).
                        flatten()[hold_back:],
                    "upper":
                        fit_scalers["train"][iteration].inverse_transform(
                            instances["ar_p"][iteration]["prediction_instances"]["i"].
                            conf_int()["upper"].values.reshape(-1, 1)).
                        flatten()[hold_back:]
                },
                "o": {
                    "DATE":
                        indices["test"]["index"].values.tolist(),
                    "source":
                        list([str(source)]
                             for i, _ in enumerate(indices["test"]["index"])),
                    "iteration":
                        list([str(iteration)]
                             for i, _ in enumerate(indices["test"]["index"])),
                    "scaler":
                        list(str(params[iteration]["ar_p"]["scaler"])
                             for i, _ in enumerate(indices["test"]["index"])),
                    "lag_order":
                        list(str(params[iteration]["ar_p"]["lag_order"])
                             for i, _ in enumerate(indices["test"]["index"])),
                    "seasonal_term":
                        list(str(params[iteration]["ar_p"]["seasonal_term"])
                             for i, _ in enumerate(indices["test"]["index"])),
                    "seasonal_order":
                        list(str(params[iteration]["ar_p"]["seasonal_order"])
                             for i, _ in enumerate(indices["test"]["index"])),
                    "trend_term":
                        list(str(params[iteration]["ar_p"]["trend_term"])
                             for i, _ in enumerate(indices["test"]["index"])),
                    "window_step":
                        list(str(i) for i, _ in enumerate(indices["test"]["index"])),
                    "real":
                        fit_scalers["test"][iteration].inverse_transform(
                            instances["ar_p"][iteration]["transformed_data"]["o"].
                            values.reshape(-1, 1)).flatten(),
                    "ex_s_prediction":
                        fit_scalers["test"][iteration].inverse_transform(
                            instances["ex_s"][iteration]["prediction_instances"]["o"].
                            values.reshape(-1, 1)).flatten(),
                    "ar_p_prediction":
                        fit_scalers["test"][iteration].inverse_transform(
                            instances["ar_p"][iteration]["prediction_instances"]["o"].
                            _predicted_mean.reshape(-1, 1)).flatten(),
                    "lower":
                        fit_scalers["test"][iteration].inverse_transform(
                            instances["ar_p"][iteration]["prediction_instances"]["o"].
                            conf_int()["lower"].values.reshape(-1, 1)).flatten(),
                    "upper":
                        fit_scalers["test"][iteration].inverse_transform(
                            instances["ar_p"][iteration]["prediction_instances"]["o"].
                            conf_int()["upper"].values.reshape(-1, 1)).flatten()
                }
            } for iteration in iterations
        }

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return initial_predictions

    def collect_predictions(self,
                            initial_predictions,
                            write_output=False):
        """
         -----------------------------------------
        ===========================================

         Generates collection of modeling predictions;

          params:
          *
          returns:
          * AR(p) model iteraation prediction elements;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Model iterations;
        iterations = self.iterations

        # # Collecting model iteration predictions;
        predictions = {
            iteration: {
                "i": pd.DataFrame.from_dict({
                    "DATE": initial_predictions[iteration]["i"]["DATE"],
                    "source": initial_predictions[iteration]["i"]["source"],
                    "iteration": initial_predictions[iteration]["i"]["iteration"],
                    "scaler": initial_predictions[iteration]["i"]["scaler"],
                    "lag_order": initial_predictions[iteration]["i"]["lag_order"],
                    "seasonal_term": initial_predictions[iteration]["i"]["seasonal_term"],
                    "seasonal_order": initial_predictions[iteration]["i"]["seasonal_order"],
                    "trend_term": initial_predictions[iteration]["i"]["trend_term"],
                    "window_step": initial_predictions[iteration]["i"]["window_step"],
                    "real": initial_predictions[iteration]["i"]["real"],
                    "ex_s_prediction":
                        initial_predictions[iteration]["i"]["ex_s_prediction"],
                    "ar_p_prediction":
                        initial_predictions[iteration]["i"]["ar_p_prediction"],
                    "lower": initial_predictions[iteration]["i"]["lower"],
                    "upper": initial_predictions[iteration]["i"]["upper"]
                }, orient="columns"),
                "o": pd.DataFrame.from_dict({
                    "DATE": initial_predictions[iteration]["o"]["DATE"],
                    "source": initial_predictions[iteration]["o"]["source"],
                    "iteration": initial_predictions[iteration]["o"]["iteration"],
                    "scaler": initial_predictions[iteration]["o"]["scaler"],
                    "lag_order": initial_predictions[iteration]["o"]["lag_order"],
                    "seasonal_term": initial_predictions[iteration]["o"]["seasonal_term"],
                    "seasonal_order": initial_predictions[iteration]["o"]["seasonal_order"],
                    "trend_term": initial_predictions[iteration]["o"]["trend_term"],
                    "window_step": initial_predictions[iteration]["o"]["window_step"],
                    "real": initial_predictions[iteration]["o"]["real"],
                    "ex_s_prediction":
                        initial_predictions[iteration]["o"]["ex_s_prediction"],
                    "ar_p_prediction":
                        initial_predictions[iteration]["o"]["ar_p_prediction"],
                    "lower": initial_predictions[iteration]["o"]["lower"],
                    "upper": initial_predictions[iteration]["o"]["upper"]
                }, orient="columns")
            } for iteration in iterations
        }

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        # Runtime for `write_output` step in `write_output` function;
        if not write_output:
            return predictions
        else:
            all_predictions = self.write_predictions(
                predictions=predictions)
            return {"predictions": predictions, "all_predictions": all_predictions}

    def write_predictions(self, predictions):
        """
         -----------------------------------------
        ===========================================

         Writes predictions output to .csv file;;

          params:
          * predictions: (dict), Dictionary containing consolidated prediction elements
           from Baseline and Primary modeling iterations;


        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Model iterations;
        iterations = self.iterations

        # All combination predictions;
        all_predictions = pd.concat([
            pd.concat([
                predictions[iteration]["i"],
                predictions[iteration]["o"]],
                ignore_index=True, axis=0)
            for iteration in iterations],
            ignore_index=True, axis=0)

        output_dir = "process_output/predictions"
        filename = "univariate_process_predictions.csv"

        all_predictions.to_csv("/".join([output_dir, filename]), index=False)

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        return (predictions, all_predictions)

    def model_information(self,
                          instances,
                          predictions,
                          write_output=False):
        """
         -----------------------------------------
        ===========================================

         Retrieves AR(p) modeling iterations information;
          params:
          * ar_p_params: (dict), Dictionary of AR iteration parameters;
          * instances: (dict), Dictionary of model instances(baseline + model);
          * predictions: (dict), Dictionary containing In-/Out-of-Sample predictions
          * source_keys: (list), List containing modeling data sources to iterate over;
          * iterations: (list), List containing model iteration names to iterate over;
          * scaler_keys: (list), List containing names of sklearn scalers to iterate over;
          * write_output: (bool), If True: Write `all_predictions to csv;`

          returnsionary containing ..
          * self.ar_models.arroots(All outside unit circle)
          * self.ar_models.llf(Log-Likelihood Function)
          * self.ar_models.aic(Akaike's Information Criterion)
          * self.ar_models.bic(Bayes' Information Criterion)
          * self.ar_models.fpe(Lutkepohl's Final Prediction Error)
          * self.ar_models.hqic(Hannan-Quinn Information Criterion)
          * self.ar_models.nobs(N Obs after lag operations);

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Model parameters;
        params = self.params

        # Model iterations;
        iterations = self.iterations

        # AR(p) model results information;
        table = {
            iteration: pd.concat([
                pd.DataFrame.from_dict(
                    {
                        "iteration":
                            str(iteration),
                        "scaler":
                            str(params[iteration]["ar_p"]["lag_order"]),
                        "lag_order":
                            params[iteration]["ar_p"]["lag_order"],
                        "season":
                            params[iteration]["ar_p"]["seasonal_order"],
                        "trend":
                            params[iteration]["ar_p"]["trend_term"],
                        "ar_roots":
                            np.where(
                                np.all(
                                    abs(instances["ar_p"][iteration]["result_instances"].
                                        roots)),
                                "All AR Roots outside of unit circle",
                                "One or more AR root outside of unit circle"
                            ),
                        "breakvar":
                            np.where(breakvar_heteroskedasticity_test(
                                instances["ar_p"][iteration]["result_instances"].resid)[1] <=
                                0.05,
                                "Ho: Presence of heteroskedasticity unlikely",
                                "Ha: Presence of heteroskedasticity likely"
                            ),
                        "normal":
                            np.where(
                                instances["ar_p"][iteration]["result_instances"].
                                test_normality()["P-value"] >= 0.05,
                                "Ha: Not Normal",
                                "Ho: Is Normal").tolist(),
                        "aic":
                            round(instances["ar_p"][iteration]
                                  ["result_instances"].aic, 5),
                        "bic":
                            round(instances["ar_p"][iteration]
                                  ["result_instances"].bic, 5),
                        "hqic":
                            round(instances["ar_p"][iteration]
                                  ["result_instances"].hqic, 5),
                        "fpe":
                            round(instances["ar_p"][iteration]
                                  ["result_instances"].fpe, 5),
                        "llf":
                            round(instances["ar_p"][iteration]
                                  ["result_instances"].llf, 5),
                        "mean_actual":
                            round(predictions[iteration]["o"]["real"].mean(),
                                  5),
                        "mean_ex_s":
                            round(predictions[iteration]["o"]["ex_s_prediction"].mean(),
                                  5),
                        "mean_ar_p":
                            round(predictions[iteration]["o"]["ar_p_prediction"].mean(),
                                  5),
                        "sigma_actual":
                            round(predictions[iteration]["o"]["real"].std(),
                                  5),
                        "sigma_ex_s":
                            round(predictions[iteration]["o"]["ex_s_prediction"].std(),
                                  5),
                        "sigma_ar_p":
                            round(predictions[iteration]["o"]["ar_p_prediction"].std(),
                                  5),
                        "median_actual":
                            round(predictions[iteration]["o"]["real"].median(),
                                  5),
                        "median_ex_s":
                            round(predictions[iteration]["o"]["ex_s_prediction"].median(),
                                  5),
                        "median_ar_p":
                            round(predictions[iteration]["o"]["ar_p_prediction"].median(),
                                  5),
                        "ex_s_i_rmse":
                            round(
                                np.sqrt(
                                    mean_squared_error(
                                        predictions[iteration]["i"]["real"],
                                        predictions[iteration]["i"]
                                        ["ex_s_prediction"])), 5),
                        "ex_s_i_mape":
                            round(
                                100 *
                                (mean_absolute_percentage_error(
                                    predictions[iteration]["i"]["real"],
                                    predictions[iteration]["i"]["ex_s_prediction"])),
                                5),
                        "ex_s_o_rmse":
                            round(
                                np.sqrt(
                                    mean_squared_error(
                                        predictions[iteration]["o"]["real"],
                                        predictions[iteration]["o"]
                                        ["ex_s_prediction"])), 5),
                        "ex_s_o_mape":
                            round(
                                100 *
                                (mean_absolute_percentage_error(
                                    predictions[iteration]["o"]["real"],
                                    predictions[iteration]["o"]["ex_s_prediction"])), 5),
                        "ar_p_i_rmse":
                            round(
                                np.sqrt(
                                    mean_squared_error(
                                        predictions[iteration]["i"]["real"],
                                        predictions[iteration]["i"]
                                        ["ar_p_prediction"])), 5),
                        "ar_p_i_mape":
                            round(
                                100 *
                                (mean_absolute_percentage_error(
                                    predictions[iteration]["i"]["real"],
                                    predictions[iteration]["i"]["ar_p_prediction"])),
                                5),
                        "ar_p_o_rmse":
                            round(
                                np.sqrt(
                                    mean_squared_error(
                                        predictions[iteration]["o"]["real"],
                                        predictions[iteration]["o"]
                                        ["ar_p_prediction"])), 5),
                        "ar_p_o_mape":
                            round(
                                100 *
                                (mean_absolute_percentage_error(
                                    predictions[iteration]["o"]["real"],
                                    predictions[iteration]["o"]["ar_p_prediction"])), 5)
                    },
                    orient="index")], axis=1) for iteration in iterations
        }

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # logging;
        # logging.info("Runtime: %s ;", runtime)

        # Runtime for `write_output` step in `write_output` function;
        if write_output:
            all_tables = self.write_information(
                table=table)
            return all_tables
        else:
            return table

    def write_information(self, table):
        """
         -----------------------------------------
        ===========================================

         Writes model information output to .csv file:
          * table: (dict), Dictionary containing model iteration information;

        ===========================================
         -----------------------------------------
        """

        # Start timing;
        # start = time.perf_counter()

        # Model iterations;
        iterations = self.iterations

        # All combination tables;
        all_tables = pd.concat([
            table[iteration]
            for iteration in iterations],
            ignore_index=False, axis=1).T.reset_index(drop=True)

        output_dir = "process_output/tables"
        filename = "univariate_process_tables.csv"

        all_tables.to_csv("/".join([output_dir, filename]), index=True)

        # Finish timing;
        # finish = time.perf_counter()

        # Runtime;
        # runtime = round(abs(start - finish), 3)

        # Logging;
        # logging.info("Runtime: %s ;", runtime)

        return all_tables

    def plot_forecasts(self,
                       predictions,
                       table,
                       truncate=1.75,
                       lines={"real": {"style": "-",
                                       "color": "grey"},
                              "ar_p_prediction": {"style": "--",
                                                  "color": "darkblue"},
                              "ex_s_prediction": {"style": "--",
                                                  "color": "orange"},
                              "interval": {"color": "lightblue"}}):
        """
         -----------------------------------------
        ===========================================

            Plots model iteration forecasts;

        ===========================================
         -----------------------------------------
         """

        # Start timing;
        # start = time.perf_counter()

        # Model parameters;
        params = self.params

        # Model iterations;
        iterations = self.iterations

        # Checking results;
        for iteration in iterations:
            table_i = table[table.iteration == iteration]
            ar_p = table_i["lag_order"].values
            source = table_i["source"].values

            # Main plot title;
            title = (
                "".join([f"Simple AR({ar_p})",
                         f" {self.window}-Day Forecasts,",
                         f"{str(source).replace('_',' ')}"])
            )

            # String-formated alpha level;
            ci_pct = "".join([str(int((1 - 0.05) * 100)), "%"])

            # Time-Series axis;
            ts_fig, ts_ax = plt.subplots(1, 1, figsize=(14, 8))

            # Truncating total time window of modeling set;
            truncate_window = int(round(truncate * self.window))

            # Appended series of prediction/plotting elements;
            t_series = pd.concat([predictions[iteration]["i"],
                                  predictions[iteration]["o"]],
                                 axis=0, ignore_index=True).reset_index(drop=True)

            # Date-indexing;
            t_series = t_series.set_index(["DATE"])

            # Real values;
            t_real = t_series["real"][-truncate_window:]

            # Modle predictions;
            t_model = t_series["ar_p_prediction"][-self.window:]

            # Baseline predictions;
            t_base = t_series["ex_s_prediction"][-self.window:]

            # Model prediction intervals;
            t_intervals = t_series[["lower", "upper"]][-self.window:]
            lower = t_intervals["lower"]
            upper = t_intervals["upper"]

            # Actual values;
            t_real.plot(ax=ts_ax,
                        ls=lines["real"]["style"],
                        color=lines["real"]["color"],
                        label="Actual")

            # Predicted values;
            t_model.plot(ax=ts_ax,
                         ls=lines["ar_p_prediction"]["style"],
                         color=lines["ar_p_prediction"]["color"],
                         label="Model Predicted")

            # Predicted values;
            t_base.plot(ax=ts_ax,
                        ls=lines["ex_s_prediction"]["style"],
                        color=lines["ex_s_prediction"]["color"],
                        label="Baseline Predicted")

            # Shading area between lower, upper interval boundaries;
            ts_ax.fill_between(t_intervals.index,
                               lower,
                               upper,
                               color=lines["interval"]["color"],
                               alpha=.1,
                               label=" ".join([ci_pct, "Confidence Interval"]))

            # Setting title;
            ts_ax.set_title(title)

            # Setting Y-Axis label;
            ts_ax.set_ylabel("STREAMS")

            # Function for rotating tick labels on x-axis only
            for tick in ts_ax.get_xticklabels():
                tick.set_rotation(45)

            # Plot legend/key;
            ts_ax.legend(loc="upper left", facecolor="snow")

            # Text elements prdataties;
            props_1 = {
                "boxstyle": "round",
                "facecolor": "snow",
                "alpha": 0.6}

            # Forecasting dates text element object
            axtxtdate = plt.subplot()

            # Prediction errors text element object
            axtxterr = plt.subplot()

            # Mean/Median values text element object
            axtxtavg = plt.subplot()

            # Dates, Errors, Means, Medians;
            datestr = "".join((
                f"{str(predictions[iteration]['o'].DATE.min())[:10]}",
                "-...-",
                f"{str(predictions[iteration]['o'].DATE.max())[:10]}"
            ))

            errstr = "".join((
                f"Exp. Smoothing Out-Of-Sample RMSE = {table_i['ex_s_o_rmse'].values}\n",
                f"Model Out-Of-Sample RMSE = {table_i['ar_p_o_rmse'].values}\n",
                f"Exp. Smoothing Out-Of-Sample MAPE = {table_i['ex_s_o_mape'].values}\n",
                f"Model Out-Of-Sample MAPE = {table_i['ar_p_o_mape'].values}"
            ))

            descstr = "".join((
                f"Actual Mean = {table_i['mean_actual'].values}\n",
                f"Exp. Smoothing Mean = {table_i['mean_ex_s'].values}\n",
                f"Auto-Regressive(p) Mean = {table_i['mean_ar_p'].values}\n",
                f"Actual Std. Deviation = {table_i['sigma_actual'].values}\n",
                f"Exp. Smoothing Std. Deviation = {table_i['sigma_ex_s'].values}\n",
                f"Auto-Regressive(p) Std. Deviation = {table_i['sigma_ar_p'].values}\n",
                f"Actual Median = {table_i['median_actual'].values}\n",
                f"Exp. Smoothing Median = {table_i['median_ex_s'].values}\n",
                f"Auto-Regressive(p) Median = {table_i['median_ar_p'].values}"
            ))

            # Place dates text box in upper left in axes coords
            axtxtdate.text(0.01,
                           0.84,
                           datestr,
                           transform=axtxtdate.transAxes,
                           fontsize=10,
                           verticalalignment="bottom",
                           bbox=props_1)
            # Place errors text box in upper left in axes coords
            axtxterr.text(0.01,
                          0.78,
                          errstr,
                          transform=axtxterr.transAxes,
                          fontsize=10,
                          verticalalignment="top",
                          bbox=props_1)
            # place values text box in upper left in axes coords
            axtxtavg.text(0.01,
                          0.68,
                          descstr,
                          transform=axtxtavg.transAxes,
                          fontsize=10,
                          verticalalignment="top",
                          bbox=props_1)

            # Plotting formalities
            sns.despine()
            # plt.tight_layout()

            # Display plot;
            # plt.show()
            # plt.savefig(f"plots/Aggregated_{s}_{self.window}DayForecastPlots")

            # Finish timing;
            # finish = time.perf_counter()

            # Runtime;
            # runtime = round(abs(start - finish), 3)
