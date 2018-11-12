/* Homework 4 solution code for Stat 448, Spring 2018 at
   University of Illinois at Urbana Champaign */
/* Results and report file were generated using SAS Studio */
data sleep;
	infile 'C:/Stat 448/sleep.csv' dlm=',';
	input species $ bodyweight brainweight nondreamingsleep dreamingsleep
		totalsleep maxlifespan gestationtime predationindex sleepexposureindex
		overalldangerindex;
	maxlife10=maxlifespan < 10;
	drop nondreamingsleep dreamingsleep;
run;
/*proc print data=sleep (obs=10);
run;*/

/*------------*/
/* Exercise 1 */
/*------------*/
ods text = "HW 4 Spring 2018";
ods text = " ";
ods text = " ";
ods text = "For the Cbar values in this homework solution, the cutoff is 0.5. The Cook's distance cutoff value is 1.";
ods text = " ";
ods text = " ";
ods text="Exercise 1 a";


/* a) Determine the best set of predictors for the model and comment on any unduly
influential points. If any extremely unduly influential points exist, remove 
them and perform selection again before choosing a final model. */
proc logistic data=sleep plots=influence(unpack);
	class predationindex sleepexposureindex overalldangerindex;
	model maxlife10 (event='0') = bodyweight brainweight totalsleep gestationtime predationindex 
		sleepexposureindex overalldangerindex / selection=forward sle=.05;
	ods select ModelBuildingSummary CBarPlot PearsonResidualPlot;
run;
ods text="1a) The best model from stepwise selection is with the predictor gestationtime. 
In the diagnostic plots, we find no unduly influential points based on the residuals 
and Cbar measures (cutoff of 0.5). Thus we will keep all observations. Note that SAS’s message 
about the validity of the model being questionable (and quasicomplete separation of points) 
is from Step 2 of the selection process, which contains overalldangerindex and gestationtime. 
The final model with only gestationtime is the problem-free choice.";

ods text = " ";
ods text = " ";
ods text="Exercise 1 b";
/* b) If any points are still too influential in your final model, remove them
   and refit. Comment on the significance of parameter estimates, what 
   Hosmer-Lemeshow's test tells us about goodness of fit, and point out any
   remaining issues with diagnostics. */
proc logistic data=sleep;
 model maxlife10 (event='0') = gestationtime / lackfit rsquare ;
 ods select GlobalTests FitStatistics LackFitChiSq;
 ods select ParameterEstimates OddsRatios;
 ods select ModelInfo ConvergenceStatus NObs;
 ods select ResponseProfile  Rsquare;
run;
ods text="1b) Now the final model is fitted with one predictor, gestation time. First, 
it is significant with a p-value of 0.004. Hence, we can conclude that gestation 
time is statistically significant in predicting the probability that a species' maximum lifespan 
will be at least 10 years. The goodness of fit test for the model has a p-value of 0.407, 
which indicates the model fit is reasonable. The r-square is 0.233 which means the model 
predicts poorly.";

ods text = " ";
ods text = " ";
ods text="Exercise 1 c";
/* c) Comment on the significance of odds ratios and interpret what the model tells 
   us about relationships between the predictors and the odds of a species' 
   maximum lifespan being at least 10 years. */
ods text="1c) The odds ratio is estimated as 1.012 so we can say for a one day increase 
in gestation time, we expect to see roughly an 1.2% increase in the odds of a species having 
maximum lifespan at least 10 years versus less than 10 years. The 95% confidence interval 
(1.004, 1.020) does not contain 1, thus the odds ratio for gestation time is statistically 
significant. Practically though, the gestation time barely increases the odds of a species 
having a maximum lifespan of 10 years.";


/*------------*/
/* Exercise 2 */
/*------------*/
ods text = " ";
ods text = " ";
ods text="Exercise 2 a";
/* a) Determine the best set of predictors for the model and comment on any unduly
influential points. If any extremely unduly influential points exist, remove 
them and perform selection again before choosing a final model. */

proc logistic data=sleep plots=influence(unpack);
title "Iteration 1";
	model maxlife10 (event='0') = bodyweight brainweight totalsleep gestationtime predationindex 
		sleepexposureindex overalldangerindex / selection=stepwise sle=0.05 sls=0.05;
	output out=sleep2 cbar=cb2;
	ods select ModelBuildingSummary CBarPlot PearsonResidualPlot;
run;
proc print data=sleep2;
 where cb2 > 0.5;
run;

proc logistic data=sleep2 plots=influence(unpack);
title "Iteration 2";
 model maxlife10 (event='0') = bodyweight brainweight totalsleep gestationtime predationindex 
  sleepexposureindex overalldangerindex / selection=stepwise sle=0.05 sls=0.05;
 where cb2 < 1;
 output out=sleep3 cbar=cb3;
 ods select ModelBuildingSummary CBarPlot PearsonResidualPlot;
run;
proc print data=sleep3;
 where cb3 > 0.5;
run;

proc logistic data=sleep3 plots=influence(unpack);
title "Iteration 3";
 model maxlife10 (event='0') = bodyweight brainweight totalsleep gestationtime predationindex 
  sleepexposureindex overalldangerindex / selection=stepwise sle=0.05 sls=0.05;
 where cb3 < 0.5;
 output out=sleep4 cbar=cb4;
 ods select ModelBuildingSummary CBarPlot PearsonResidualPlot;
run;
proc print data=sleep4;
 where cb4 > 0.5;
run;

proc logistic data=sleep4 plots=influence(unpack);
title "Iteration 4";
 model maxlife10 (event='0') = bodyweight brainweight totalsleep gestationtime predationindex 
  sleepexposureindex overalldangerindex / selection=stepwise sle=0.05 sls=0.05;
 where cb4 < 0.5;
 output out=sleep5 cbar=cb5;
 ods select ModelBuildingSummary CBarPlot PearsonResidualPlot;
run;
title;
proc print data=sleep5;
 where cb5 > 0.5;
run;
 ods text="2a) Treating the index variables as continuous, we needed to do several model 
fittings and re-fittings due to influential points. The diagnostic plots (residuals and Cbar)
as well as data print outs of the Cbar values guided us to remove 3 observations due to 
influence. These observations were 44, 62, and 59 were removed one by one. Thus the final 
model we chose contained sleepexposureindex as the only predictor.";

ods text = " ";
ods text = " ";
ods text="Exercise 2 b";
/* b) If any points are still too influential in your final model, remove them
   and refit. Comment on the significance of parameter estimates, what 
   Hosmer-Lemeshow's test tells us about goodness of fit, and point out any
   remaining issues with diagnostics. */
proc logistic data=sleep5 ;
 model maxlife10 (event='0') = sleepexposureindex/ lackfit rsquare;
 ods select GlobalTests FitStatistics LackFitChiSq;
 ods select ParameterEstimates OddsRatios;
 ods select ModelInfo ConvergenceStatus NObs;
 ods select ResponseProfile  Rsquare ;
run;
ods text="2b) Now, the final model is fitted with one predictor, sleep exposure index. 
Sleep exposure index is statistically significant (p-value 0.004) meaning its estimated 
coefficient is far from 0. Thus it aids in predicting whether the maximum lifespan of a 
species will be at least 10 years. The goodness of fit test for the model has a p-value 
of 0.574, which indicates the model fit is reasonable. The r-square is 0.323 which means 
the model predicts poorly, but better than the model in Exercise 1 which treated index 
variables including sleep exposure index as a categorical variable.";

ods text = " ";
ods text = " ";
ods text="Exercise 2 c";
/* c) Comment on the signicance of odds ratios and interpret what the model tells 
   us about relationships between the predictors and the odds of a species' 
   maximum lifespan being at least 10 years. */
ods text="2c) The odds ratio is estimated as 2.890 so we can say for a one-unit increase 
in sleep exposure index, we expect to see roughly an increase in the odds of a species having 
maximum lifespan at least 10 years of 2.89 times the odds of a species having maximum lifespan 
less than 10 years.The 95% confidence interval (1.597, 5.231) does not contain 1, thus the odds 
ratio for sleep exposure index (treated as continuous) is statistically significant. This 
reflects a strong practical increase in lifespan when the sleep exposure index grows.";




/*------------*/
/* Exercise 3 */
/*------------*/
ods text = " ";
ods text = " ";
ods text="Exercise 3 a";
/* a) Determine the best set of predictors for the model (accounting for overdispersion
   if necessary) and comment on any unduly influential points. If any extremely unduly 
   influential points exist, remove them and refit before proceeding with model selection. 
   Comment on what type 1 and type 3 analyses tell us about predictors you may want to 
   keep or remove from the model. */
proc genmod data=sleep plots=cooksd; 
	model gestationtime = bodyweight brainweight predationindex 
		sleepexposureindex / dist=poisson ;
	output out=sleepII CooksD=cdII;
	ods select ModelFit ModelInfo ;
run; 
/*since above model shows overdispersion, we must use 
overdispersion Poisson model then check diagnostics*/

proc genmod data=sleep plots=cooksd;
title "Iteration 1";
	model gestationtime = bodyweight brainweight predationindex 
		sleepexposureindex / dist=poisson scale=deviance ;
	output out=sleepII CooksD=cdII;
	ods select ModelFit ModelInfo CooksDPlot;
run; 
proc print data=sleepII;
 where cdII > 1;
run;

proc genmod data=sleepII plots=cooksd;
title "Iteration 2";
	model gestationtime = bodyweight brainweight predationindex 
		sleepexposureindex / dist=poisson scale=deviance ;
	where cdII < 50;
	output out=sleepIII CooksD=cdIII;
	ods select  CooksDPlot;
run;
proc print data=sleepIII;
 where cdIII > 1;
run;

proc genmod data=sleepIII plots=cooksd;
title "Iteration 3";
	model gestationtime = bodyweight brainweight predationindex
		sleepexposureindex/dist=poisson scale=deviance type1 type3;
	where cdIII <200;
	output out=sleepIV CooksD=cdIV;
	ods select CooksDPlot;
run;
proc print data=sleepIV;
 where cdIV > 1;
run;

proc genmod data=sleepIV plots=cooksd;
title "Iteration 4";
	model gestationtime = bodyweight brainweight predationindex
		sleepexposureindex/dist=poisson scale=deviance type1 type3;
	where cdIV <1;
	output out=sleepV CooksD=cdV;
	ods select ModelFit ModelInfo CooksDPlot;
run;
proc print data=sleepV;
 where cdV > 1;
run;

*the 4 term model;
ods text = " ";
ods text = " ";
ods text="4-term model";
proc genmod data=sleepV plots=cooksd;
title "Model Bo,Br,Pr,Sl";
	model gestationtime =  bodyweight brainweight predationindex
		sleepexposureindex / dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;

/*
*trying various 3-term models;
ods text="3-term models";
proc genmod data=sleepV plots=cooksd;
title "Iteration Br,Pr,Sl";
	model gestationtime =  brainweight predationindex
		sleepexposureindex / dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;
proc genmod data=sleepV plots=cooksd;
title "Iteration Bo,Pr,Sl";
	model gestationtime = bodyweight  predationindex
		sleepexposureindex / dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;
*/
ods text = " ";
ods text = " ";
ods text="3-term model";

proc genmod data=sleepV plots=cooksd;
title "Model Bo,Br,Sl";
	model gestationtime = bodyweight brainweight 
		sleepexposureindex / dist=poisson scale=deviance type1 type3;
	output out=final STDRESDEV= resids pred=pg;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;

/*
proc genmod data=sleepV plots=cooksd;
title "Iteration Bo,Br,Pr";
	model gestationtime = bodyweight brainweight predationindex
		/ dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;

*trying various 2-term models;
ods text="2-term models";
proc genmod data=sleepV plots=cooksd;
title "Iteration Bo,Br";
	model gestationtime = bodyweight brainweight / dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;
proc genmod data=sleepV plots=cooksd;
title "Iteration Bo,Pr";
	model gestationtime = bodyweight  predationindex / dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;
proc genmod data=sleepV plots=cooksd;
title "Iteration Bo,Sl";
	model gestationtime = bodyweight sleepexposureindex / dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;
proc genmod data=sleepV plots=cooksd;
title "Iteration Br,Pr";
	model gestationtime = brainweight predationindex / dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;
proc genmod data=sleepV plots=cooksd;
title "Iteration Br,Sl";
	model gestationtime = brainweight sleepexposureindex / dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;
proc genmod data=sleepV plots=cooksd;
title "Iteration Pr,Sl";
	model gestationtime = predationindex sleepexposureindex / dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo Type1 ModelAnova CooksDPlot;
run;
*/
title;
ods text="3a) Note that we observe overdispersion since the scale value 
(scaled deviance 53.101) is much larger than 1 when all 4 predictors are 
in the model. When an overdispersed Poisson log-linear model is fitted 
using the four predictors we needed to do several model fittings and 
re-fittings due to influential points. The Cook's Distance plot as well as data 
print outs of the Cook's D values guided us to remove 3 observations due to 
influence. These observations were 5, 1, and 32 which were removed one by 
one. After removing those influential points, we refit the model and
choose the best set of predictors. To do this we should fit several 
competing models and compare their information criterion values 
(AIC, AICc, BIC). The two best models are: the 4 predictor model with
BIC value of 2212.596 and the 3 predictor model 
(bodyweight, brainweight, sleepexposureindex) with BIC value of 2256.158.
We would choose the 3 predictor model since its BIC is not much different 
from the full model and is simpler (parsimonious). Based on type 1 and 
type 3 analysis results, we can keep these 3 significant (at 5% level) 
predictors.";


ods text = " ";
ods text = " ";
ods text="Exercise 3 b";
/* b) Based on the results from part a, determine what terms should be retained 
   for the final model. After choosing the final terms, remove any points that are
   still unduly influential and refit. Leave no points with Cook's distance greater 
   than 1 in your data. */
proc genmod data=sleepV plots=(cooksd stdreschi stdresdev);
	model gestationtime = bodyweight brainweight
		sleepexposureindex / dist=poisson scale=deviance type1 type3;
	ods select ModelFit ModelInfo ParameterEstimates Type1 ModelAnova 
		DiagnosticPlot;
run;
proc sgplot data=final;
	scatter y=resids x=pg;
run;
ods text="3b) We fit the log-linear model for gestation time using 
bodyweight, brainweight and sleepexposureindex as predictors. From 
the Cook's Distance and residuals plots, we do not have any 
influential observations of concern. So there is no need to remove 
any points. We can also see that there is no trend in the residuals 
vs. predicted values, so the distribution and link choices are reasonable." ;

ods text = " ";
ods text = " ";
ods text="Exercise 3 c";
/* c) Comment on the significance of parameter estimates and point out any remaining
   issues with diagnostics in your final model. Interpret what the parameter estimates 
   tell us about how the predictors are related to expected gestation time. */
ods text="3c) From type 1 and type 3 analysis tables, we can see that 
all predictors are significant (at 5% level) in the model. The parameter estimates 
are -0.002, 0.0034 and 0.200 for bodyweight, brainweight and 
sleepexposureindex respectively. The bodyweight has negative value 
so it implies that there's an expected decrease in log gestation 
time for a one-unit increase in bodyweight is -0.002 while holding 
the other predictors constant. The expected gestation time would 
decrease by e^-0.002=0.998 times. The remaining parameter estimates 
are positive indicating an expected increase in log gestation time. 
For a one-unit increase in brainweight, the expected log count of 
gestation time increases by .0034 (multiplicative increase of 
e^0.0034=1.003 in gestation time) holding all other predictors constant.
For a one-unit increase in sleepexplosure index, the expected log 
count of gestation time increases in by 0.200 (multiplicative 
increase of e^0.200=1.22). There are no other conerns in the diagnostics.";
