/* Solution code for Homework 2 in Stat 448, University of Illinois, Spring 2018 */
ods html close; 
options nodate nonumber leftmargin=1in rightmargin=1in;
title;
ods escapechar="~";
ods rtf file='C:\Stat 448\HW2_Sp18_Solution.rtf' startpage=no;
ods noproctitle; 
data haireyes;
	input eyecolor $ haircolor $ count;
	datalines;
	light fair 688
	light medium 584
	light dark 188
	medium fair 343
	medium medium 909
	medium dark 412
	dark fair 98
	dark medium 403
	dark dark 681
	;
data heartbpchol;
	set sashelp.heart;
	where status='Alive' and AgeCHDdiag ne . and Chol_Status ne ' ';
	if BP_Status = 'Optimal' then index = 1;
	if BP_Status = 'Optimal' and Chol_Status = 'Borderline' then index = 2;
	if BP_Status = 'Optimal' and Chol_Status = 'High' then index = 3;
	if BP_Status = 'Normal' then index = 4;
	if BP_Status = 'Normal' and Chol_Status = 'Borderline' then index = 5;
	if BP_Status = 'Normal' and Chol_Status = 'High' then index = 6;
	if BP_Status = 'High' then index = 7;
	if BP_Status = 'High' and Chol_Status = 'Borderline' then index = 8;
	if BP_Status = 'High' and Chol_Status = 'High' then index = 9;	
	keep Cholesterol BP_Status Chol_Status index;
	proc sort;
		by index;
run;
ods title "Exercise 1";
/* Exercise 1 */
proc freq data=haireyes order=data;
	table eyecolor*haircolor/chisq expected nocol nopercent;
	weight count;
	ods select CrossTabFreqs ChiSq;
run;
ods text="";
ods text = "~n~n

1a) Comparing observed and expected counts or looking for difference in row percentages within columns can provide evidence association. There is a noticeably larger than expected number of light eye color with fair hair, medium eye color with medium hair color, and dark eyes with dark hair. There are slightly lower than expected counts for medium values paired with non-medium values. Finally, there are far fewer than expected observations with dark hair and light eyes and fair hair with dark eyes. This suggests an association. In particular, observations with similar levels of hair darkness tend to occur more often and observations with greater difference in darkness of hair and eye color tend to happen less often. This suggests a positive association since both variables are ordinal.

~n~n

1b) The sample size is large enough to look at asymptotic chi-square tests and the variables are ordinal, so we should look at Pearson's chi-square, the likelihood ratio chi-square, and the Mantel-Haenszel Test. Pearson and Likelihood Ratio are both highly significant, indicating there is a statistically significant association. Mantel-Haenszel is also statistically significant, indicating a significant linear trend. Based on these results and the trend noticed in part, we conclude there is a statistically significant positive association between eye color and hair color. Those with darker eyes are more likely to have dark hair; those with lighter eyes are more likely to have fairer hair; and those with medium eyes are more likely to have medium hair. Those with one characteristic darker and one lighter are much less likely.
~n~n";

ods title "Exercise 2";
/* Exercise 2*/
/* If using SAS Studio, we will need to create a new data set for the 
   desired subset and perform the analysis on that new data. The following 
   comment contains code that would accomplish this. */
/* 
data haireyesMM;
	input eyecolor $ haircolor $ count;
	datalines;
	light fair 688
	light dark 188
	dark fair 98
	dark dark 681
	;
run;
proc freq data=haireyesMM order=data;
	table eyecolor*haircolor/chisq expected nocol nopercent riskdiff;
	weight count;
	ods select CrossTabFreqs ChiSq RiskDiffCol1;
run;
*/
proc freq data=haireyes order=data;
	table eyecolor*haircolor/chisq expected nocol nopercent riskdiff;
	weight count;
	/* following where statement should work, and does work in regular SAS,
	   but doesn't in SAS Studio */
	where eyecolor ne 'medium' and haircolor ne 'medium';
	ods select CrossTabFreqs ChiSq RiskDiffCol1;
run;
ods text="";
ods text="~n~n

2a) Just as before, light-fair and dark-dark occur mush more often than expected, suggesting a positive association.

~n~n

2b) The sample is still large enough for asymptotic tests and the variables are ordinal, so Pearson's chi-square, Likelihood Ratio chi-square, and Mantel-Haenszel are all appropriate. Each is highly significant, indicating there is association and a linear trend in association. The phi coefficient (and Cramer's V) indicate that the association is positive with a moderate magnitude. Just as in exercise 1, similar darknesses of hair and eye color happen more frequently than expected due to chance, and those with different darkness values happen less likely.

~n~n

2c) The question of interest is about fair hair, so column 1 can be used to directly answer the question. The risk for indiviuals with light eyes to have fair hair is estimated to be .7854, and for those with dark eyes to have fair hair is estimated to be .1258. This gives an estimated difference of risks of .6596. The confidence interval for this difference is completely positive and pretty far from 0, so we conclude that those with light eyes are much more likely to have fair hair than those with dark eyes.

~n~n";

ods title "Exercise 3";
/* Exercise 3*/
proc freq data=heartbpchol order=data;
	table BP_Status*Chol_Status/chisq expected nocol nopercent riskdiff;
	ods select CrossTabFreqs ChiSq;
run;
ods text="";
ods text="
~n~n

3a) We can look at either the expected counts or the row percentsges to see that Optimal and Normal blood pressures occur more often with Desirable cholesterol level, Borderline cholesterol level are consistent with expected values in all three groups, High blood pressure occurs less often with Desirable cholesterol, High cholesterol and High blood pressure happen more often together, and High cholesterol happens less often with Optimal or Normal blood pressure. The differences are noticeable, but not particularly large, with the exception of High blood pressure for the Desirable and High cholesterol levels. This suggests there is a positive association, but it may not be very strong in general.

~n~n

3b) The variables are ordinal and data is large enough for asymptotic tests. The p-values for Pearson's chi-square and the Likelihood Ratio chi-square are both a little less than .02 so these are significant at a .05 level and indicate there is an association. The Mantel-Haenszel test is even more significant with a p-value of .0016, indicating a pretty noticeable linear trend. Higher cholesterol levels tend to be associated with higher blood pressures.
~n~n~n~n~n~n~n";

ods title "Exercise 4";
/* Exercise 4*/
proc anova data=heartbpchol;
	class BP_Status;
	model Cholesterol = BP_Status;
	means BP_Status/ hovtest tukey cldiff;
	ods select HOVFTest ModelANOVA OverallANOVA FitStatistics CLDiffs;
run;quit;
ods text="";
ods text="~n~n

4a) Since this is a one-way analysis, Levene's test for homogeneity of variance should be performed. The p-value of 0.5719 is highly insignificant so an equal variance assumption is fine here. The analysis of variance table has a p-value of 0.0014, so the model is statistically significant. More of the variation in cholesterol can be described by blood pressure level than expected due to chance. The r-squared value of .0242, however, is very small. Only 2.42% of the variation in cholesterol can be described by the blood pressure levels.

~n~n

As an added note, r-square will tend to go smaller for when a categorical predictor has been based on cutoffs for a continuous variable. If there is a linear relationship between the two continuous variables, we would expect the response just to the left of the cutoff to be pretty close to the expected response just to the right of the cutoff, but the observations would be in different groups. The reduced granularity in the predictors from binning tends to flatten out the predictions and reduce explanation of variation.

~n~n

4b) The best test to use for testing all pairwise comparisons is Tukey's test. Expected cholesterol levels are significantly different for high and normal blood pressures and for high and optimal blood pressures. We expect individuals with high blood pressures to have cholesterol levels 11.54 higher than those with normal blood pressures on average and 18.65 higher than those with optimal blood pressure on average. The difference between normal and optimal blood pressure groups is not significant.";

ods rtf close;
