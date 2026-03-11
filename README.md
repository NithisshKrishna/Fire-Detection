# INTRODUCTION

Currently, fire is detected using devices like sensors, smoke detectors.
These devices have a lot of disadvantages.
Sensors:

● Uses radioactive materials
● Extremely sensitive – False alarms are common
● Batteries have to be charged often

Smoke Detectors :

● Circuit Malfunctions
● Extremely sensitive to dust and air particles - regular maintenance
is needed.
● Need high current to function

Hence, a new method to detect fire without as many disadvantages is
essential. Moreover, detecting fire while fire is still small, is essential. Our
Machine Learning Model takes videos as input and alerts if it has detected fire.
The model has been trained with about 4320 images, so it detects all kinds of fire like forest fires and household fires, with very less probability of false
positives.

Traditionally, fire is extinguished using water, or air. Both of these
extinguishing methods require fire fighters to be in direct contact with the
flames, which is usually life threatening.
Recently, there are studies on whether sound waves can be used to
extinguish fire. 

Though it hasn’t been used yet, experiments have been
conducted to check what kind of sound waves can be used to extinguish
different kinds of flames - flames caused by gasoline, kerosene etc. We use the
results of one such experiment to train our model. The model predicts if the
given fire can be extinguished with a certain sound wave, when we have certain
features of the sound wave and the fire.
