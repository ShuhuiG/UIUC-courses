/* Homework 6 solution code for Stat 448, Spring 2018 at
   University of Illinois at Urbana Champaign */
ods html close; 
options nodate nonumber  leftmargin=1in rightmargin=1in;* topmargin=.5in bottommargin=1in;
title;
ods escapechar="~";
ods rtf file='C:\Stat 448\HW6_Sp18_Solution.rtf' startpage=no;
ods noproctitle; 
/* Exercise 1-2 */
/* The raw data in glass.data is from the UCI Machine Learning Repository. 
The data, variables and original source are described on the following URL
http://archive.ics.uci.edu/ml/datasets/Glass+Identification
*/
data glassid;
	infile "C:\Stat 448\glass.data" dlm=',' missover;
	input id RI Na Mg Al Si K Ca Ba Fe type;
	groupedtype = "buildingwindow";
	if type in(3,4) then groupedtype="vehiclewindow";
	if type in(5,6) then groupedtype="glassware";
	if type = 7 then groupedtype="headlamps";
	drop id type;
run;

ods title "Exercise 1";

ods text = "1a) To choose the optimal number of clusters, we see CCC, Pseudo F, and Pseudo t^2 statistics. First, the CCC plot has a peak at 1 and 11, and the Pseudo F plot shows a peak at around 2 and 11. In terms of Pseudo t^2, we find deep points at 3, 6, 9, and 11. The three criteria commonly say 11 clusters will be the best choice. However, note that CCC shows negative values for each number of clusters. It implies that it is hard to find clear separation among clusters in our dataset. As the next reference, the dendrogram shows that somewhere around 11 clusters could be okay, but still all clusters are close to each other. Based on diagnostics, we choose 11 clusters to be the number of clusters and continue further analysis, but negative CCC values and dendrogram imply that this dataset do not show clear separation among clusters.";

proc cluster data=glassid method=average std ccc pseudo print=15 plots(maxpoints=250);
	var Na--Fe;
	copy RI groupedtype;
	ods select CccPsfAndPsTSqPlot;
run;

proc tree n=11 out=glassclusters;
	copy RI--groupedtype;
run;

ods text = "1b) A frequency table gives the cross-cluster frequencies between 11 clusters and the original 4 glass groups. We can see that three of the glass groups, buildingwindow, glassware, and vehiclewindow, are mostly grouped to cluster 1. The glass type headlamp is mostly in cluster 2. All other clusters include a few from each glass types. Thus, we can say that 11 clusters we choose from part (a) do not match with the glass types very well. However, we can at least infer that buildingwindow, glassware, vehiclewindow have similar chemical composition (which would be a feature of glasses in cluster 1). At the same time, headlamps would have a different chemical composition compared to other three.";
proc freq data=glassclusters;
	table groupedtype*cluster/ norow nocol nopercent;
run;
ods rtf startpage=now;
ods title "Exercise 2";
ods text = "2a) From the frequency table in Exercise 1 (b), we see that only cluster 1 and 2 have more than 5 observations. So we will perform ANOVA for refractive index as a function of two clusters. Indeed, it would be an analysis to compare the means of refractive index between two groups, cluster 1 and 2. First, we see that Levene’s test for homogeneity of variance has p-value larger than .05. It implies that we can continue ANOVA and its output is valid. Second, we see that model is significant with p-value less than .001, thus we can conclude that two groups have significant difference in refractive index. Lastly, ANOVA model can explain 7.65% of total variation in refractive index, which seems a bit low.";

proc anova data=glassclusters;
	class cluster;
	model ri=cluster;
	means cluster/hovtest tukey cldiff;
	where cluster <3;
	ods select OverallANOVA ModelANOVA FitStatistics HOVFTest CLDiffs;
run;
ods text = "~n~n2b) Tukey’s pairwise comparison result shows that there is significant difference between cluster 1 and 2 in terms of refractive index, and cluster 1 has higher mean than cluster 2 has. However, as we see in part (a), its R^2 is pretty low, less than 10 %. It implies that this model is not very useful to predict refractive index.";

/* Exercise 3-4 */
/* The raw data in wine.txt is from the UCI Machine Learning Repository. The data, variables and original source are described on the following URL
http://archive.ics.uci.edu/ml/datasets/Glass+Identification

The data set and original variables are described on http://archive.ics.uci.edu/ml/datasets/Wine, and additional information about the wine data can be found at: 
http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names
*/
data wine;
	infile 'C:\Stat 448\wine.txt' dlm=',';
	input alcohol malic_acid ash alcalinity_ash magnesium total_phenols flavanoids nonflavanoid_phenols proanthocyanins color hue od280_od315 proline;
run;
ods rtf startpage=now;
ods title "Exercise 3";

ods text = "~n~n3a) We start by performing a principal component analysis on the wine characteristics and keeping the first 2 components.";

* Exercise 3a;
proc princomp data=wine n=2 out=pcwine plot=score;
	var malic_acid--proline;
	id alcohol;
	ods select ScorePlot;
run;
ods text = "The score plot shows noticeable difference in the location of the alcohol types, indicating noticeable differences in the the most prominant features of the alcohols. Alcohol type 1 tends to have higher values for both component 1 and component 2. Alcohol type 2 tends to have roughly average values for component 1 and lower than average values for component 2. Alcohol 3 tends to have lower than avergae values for component 1 and higher than average values for component 2.";

* 3b;
ods text = "~n~n3b) Next we cluster the observations based on these two principal components using average linkage.";
proc cluster data=pcwine method=average outtree=clwine ccc pseudo;
	var prin1 prin2;
	copy alcohol;
	ods select CccPsfAndPsTSqPlot;
run;
ods text = "The CCC and pseudo F plots both show high points at 3 clusters, and pseudo t-squared has a low point at 3 with a big jump at 2. These all suggest 3 as the best choice of number of clusters. In the dendrogram, we might consider 4 but notice that one of the 4 clusters would only have 1 observation. Three clusters omitting that isolated point would be a good choice based on the dendrogram.";
proc tree data=clwine ncl=3 out=cltreewine;
	copy alcohol;
run;

* 3c;
ods text = "3c) Frequency analysis will help determine how well the alcohol types are separated by these clusters.";
proc freq data=cltreewine;
	table cluster*alcohol/ norow nocol nopercent;
run;
ods text = "Alcohol 3 is very well separated based on the frequency table. Nearly all of the alcohol 3 observations are in cluster 1. Only two alcohol 3 observations are in another cluster, and only 1 observation from a different alcohol type is in cluster 1. ~n~n
Separation of alcohol types 1 and 2 are not very good. Most of alcohol 2 is in cluster 3, but roughly a third of alcohol 2 observations are grouped with all of the alcohol 1 observations in cluster 2.";
ods rtf startpage=now;
ods title "Exercise 4";

* Exercise 4a;
ods text = "4a) Results for a stepwise selection follow. Based on these results, we will want to use all of our predictors for a discriminant analysis for alcohol type.";
proc stepdisc data=wine;
	class alcohol;
	var malic_acid--proline;
	ods select Summary;
run;
ods rtf startpage=now;
ods text = "4b) Results for discrimination based on all predictors follow. There is a noticeable, but not huge, difference in the number of observations from each group. We will use proportional priors, but equal priors could also be reasonable here.~n~n
The Chi-square test is also highly significant, indicating that we need to account for differences of covariance across groups, and use quadratic discriminant analysis. The MANOVA tests are all highly significant, indicating that there is noticeable difference in values for at least some of the predictors for at least some of the groups. Therefore, we should stand a chance of discriminating between some groups based on some predictors in our model.";

* 4 b and c;
proc discrim data=wine manova pool=test crossvalidate;
	class alcohol;
	var malic_acid--proline;
	priors prop;
	ods select MultStat ChiSq ClassifiedCrossVal ErrorCrossVal;
run;
ods text = "4c) A look at the cross-validation error estimates shows that this model worked quite well, with an overall error rate under 3% and all individual error rates under 5%. All of the type 3 alcohols are correctly classified, only two type 1 alcohols are misclassified as type 2 and three type 2 misclassified as type 1.~n~n
The classification is much more successful than clustering based on the first two principal components at identifying the three groups. This should not be too surprising given that the discrimination used all variables in the data rather than just the two most prominent underlying features, and discriminant analysis trains based on known groups while clustering finds groups without using any knowledge of existing classifications.";

ods rtf close;

