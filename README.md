# Multi-modal Fusion for Estimating Fatigue <br> in Exercise Participants
This repo contains the data and programs for a fatigue estimation module.

## Dependencies
Please refer to https://github.com/ditoec/openface2_ros for installing OpenFace and its ROS Wrapper.

## Data
data/logs contains the log files of all participants. <br>
data/Study 1 - Form 2.csv contains the self-reported RPE values. <br>

## Data Visualization
script/plot_rpe.py plots the RPE values for all participants and stores in plots. <br>
script/plot_fau.py plots the FAU values from a sample fau.txt within processed_data. <br>

## Preprocessing
1. script/parse_log.py parses all log files in data/logs and stores in processed_data. <br>
2. src/quori_openface2_rosbag.cpp parses a rosbag (no need to run rosplay) and stores the fau data in processed_data. <br>
3. script/combine_features.py processes all data in processed_data and store the features and labels in features. Plots of slopes are generated in slope_plots. <br>

## Modeling
Different ML models are compared:
1. Autoregressive Moving Average (script/run_auto_reg_mov_avg.py)
2. Linear Regression (script/run_linear_regression.py)
3. Prophet (script/run_prophet.py)
4. Random Forest Regression (script/run_random_forest.py)
5. Support Vector Regression (script/run_svr.py)

## Results
| Model                    | MSE (RPE) | MSE (Rate) |
|--------------------------|----------:|----------:|
| Additive Model           |      8.30 |     24.14 |
| ARMA                     |      5.38 |      2.96 |
| Linear Regression        |      5.68 | **1.27**  |
| Random Forest Regression |      6.68 |      2.21 |
| Support Vector Regression| **5.20**  |      3.19 |

## References
1. Brown, L., Kerwin, R., and Howard, A. M. (2013, October). Applying behavioral strategies for student engagement using a robotic educational agent. In 2013 IEEE international conference on systems, man, and cybernetics (pp. 4360-4365). IEEE.
2. Martin K. Ross, Frank Broz, Lynne Baillie. "Towards an Adaptive Robot for Sports and Rehabilitation Coaching." CoRR, abs/1909.08052 (2019). [Online]. Available: http://arxiv.org/abs/1909.08052
3. R. Kaushik and R. Simmons, “Perception of Emotion in Torso and Arm Movements on Humanoid Robot Quori,” in Companion of the 2021 ACM/IEEE International Conference on Human-Robot Interaction, 2021, pp. 62–66.
4. R. Kaushik and R. Simmons, "Affective Robot Behavior Improves Learning in a Sorting Game," 2022 31st IEEE International Conference on Robot and Human Interactive Communication (RO-MAN), Napoli, Italy, 2022, pp. 436-441, doi: 10.1109/RO-MAN53752.2022.9900654.
5. Andrew Specian, Ross Mead, Simon Kim, Maja J. Mataric, Mark Yim. "Quori: A Community-Informed Design of a Socially Interactive Humanoid Robot." CoRR, abs/2109.00662 (2021).
6. T. Baltrušaitis, P. Robinson and L. -P. Morency, "OpenFace: An open source facial behavior analysis toolkit," 2016 IEEE Winter Conference on Applications of Computer Vision (WACV), Lake Placid, NY, USA, 2016, pp. 1-10, doi: 10.1109/WACV.2016.7477553.
7. Hjortsjö CH Man's face and mimic language. free download: Carl-Herman Hjortsjö, Man's face and mimic language". 1969.
8. Ekman P, Friesen WV, Hager JC. Facial Action Coding System: The Manual on CD ROM. Salt Lake City: A Human Face. 2002.
9. R. Kaushik and R. Simmons, “Early Prediction of Student Engagement-Related Events from Facial and Contextual Features,” in International Conference on Social Robotics, Springer, 2021, pp. 308–318.
10. gTTS. Google Text-to-Speech (gTTS): Python library and CLI tool. Retrieved from http://gtts.readthedocs.org/. 2014.
11. Taylor SJ, Letham B. 2017. Forecasting at scale. PeerJ Preprints 5:e3190v2 https://doi.org/10.7287/peerj.preprints.3190v2
12. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
13. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013.
14. Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.
