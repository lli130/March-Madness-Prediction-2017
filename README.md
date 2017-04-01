# March-Madness-Prediction-2017 
###################
#overivew
#################
#used  analytical techniques to predict the winners of each game in the 2017 March Madness tournament
#removed some variables such as the scores of the game from the tournament, number of overtime appearances in the tournament and pre-season poll
#also extrapolated data from BasketballReference.com 
#obtained advanced statistical variables such as Efficient Field Goal percentage, Turnovers over per 100 percentage and many others 
#After merging the datasets together, another round of the cleaning the data and implemented PCA, to combine the variables into 7 
#After implementing the PCA analysis we used 4 different models to compare and contrasts, including KNN, GB, RF, and SVM

#new added statistics:
##Four Factors: 
###1 -Shooting (eFG%) 40%: FG% adjusted for extra point received on a 3 point shot   
###2 -Turnovers (TOV%) 25%: Number of times a team turnovers the ball per 100 possessions     
###3 -Rebounding (ORB% and DRB%) 20%: What percentage of available rebounds a team receives（data limitation）     
###4 -Free Throws (FT/FGA) 15%: How often a team is able to get to the FT line
###5 Simple Rating System: Works like PageRank, attributes a rating to each team based on their point differential in each game and is summed with the point differential of each team (link below explains more in detail).
