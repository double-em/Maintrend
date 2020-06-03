# Maintrend
![Travis (.com)](https://img.shields.io/travis/com/double-em/Maintrend?label=Travis%20CI%20Tests)<br>
![Predictor build](https://github.com/Zxited/TrendlogDataApp/workflows/Predictor%20build/badge.svg?branch=master)<br>
![Serving API Build and Publish to GKE](https://github.com/Zxited/Maintrend/workflows/Serving%20API%20Build%20and%20Publish%20to%20GKE/badge.svg?branch=master)<br>
![Uptime Robot ratio (30 days)](https://img.shields.io/uptimerobot/ratio/m784993822-a76d37ac3e6259c2a679aebb?label=Serving%20API)
![Uptime Robot ratio (30 days)](https://img.shields.io/uptimerobot/ratio/m784993836-849c5728c21c6df110e3e605?label=Predictor%20Service)

Maintrend is a maintenance date predictor developed as a school project for my Computer Science test.

The predictor uses a RNN architecture with LSTM cells. The mean absolute loss for never before seen data is around 0.3 days with a maximum absolute loss at around 0.8 days.
