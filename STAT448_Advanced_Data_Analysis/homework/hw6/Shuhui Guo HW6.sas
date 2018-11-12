/* Exercise 1-2 */
/* The raw data in glass.data is from the UCI Machine Learning Repository. 
The data, variables and original source are described on the following URL
http://archive.ics.uci.edu/ml/datasets/Glass+Identification
*/
data glassid;
	*infile "C:\Stat 448\glass.data" dlm=',' missover;
	infile '/home/clever_g0/Stat 448/glass.data' dlm=',' missover;
	input id RI Na Mg Al Si K Ca Ba Fe type;
	groupedtype = "buildingwindow";
	if type in(3,4) then groupedtype="vehiclewindow";
	if type in(5,6) then groupedtype="glassware";
	if type = 7 then groupedtype="headlamps";
	drop id type;
run;
/* 1a */
proc cluster data=glassid method=average ccc std pseudo print=15 plots (maxpoints=214);*=den(height=rsq);
  var Na--Fe;
  copy RI groupedtype;
  ods select Dendrogram CccPsfAndPsTSqPlot;
run;
proc tree noprint ncl=11 out=outs1;
   copy RI groupedtype Na--Fe;
run;
proc sort data=outs1;
 by cluster;
run;
*proc print data=outs1;
*  where cluster=1 or cluster=2;
*run;
/* 1b */
proc freq data=outs1;
  tables cluster*groupedtype/ nopercent norow nocol;
run;
/* 2ab */
proc anova data=outs1;
  class cluster;
  model RI=cluster;
  where cluster=1 or cluster=2;
  means cluster/ hovtest cldiff tukey;
  ods select OverallANOVA FitStatistics CLDiffs HOVFTest;
run;
* nonparametric tests to avoid normality assumption;
* use the Dwass, Steel, Critchlow-Fligner multiple comparison procedure for multiple comparisons in nonparametric 1-way ANOVA;
*proc npar1way data=outs1 wilcoxon;
*	class cluster;
*  	var RI;
*  	where cluster=1 or cluster=2;
*  	ods exclude KruskalWallisTest;
*run;
/* Exercise 3-4 */
/* The raw data in wine.txt is from the UCI Machine Learning Repository. The data, variables and original source are described on the following URL
http://archive.ics.uci.edu/ml/datasets/Glass+Identification

The data set and original variables are described on http://archive.ics.uci.edu/ml/datasets/Wine, and additional information about the wine data can be found at: 
http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names
*/
/* 3a */
data wine;
	*infile 'C:\Stat 448\wine.txt' dlm=',';
    infile '/home/clever_g0/Stat 448/wine.txt' dlm=',';
	input alcohol malic_acid ash alcalinity_ash magnesium total_phenols flavanoids nonflavanoid_phenols proanthocyanins color hue od280_od315 proline;
run;
proc princomp data=wine n=2 out=outs2;
  var malic_acid--proline;
  ods exclude SimpleStatistics;
run;
proc sgplot data=outs2;
  scatter y=prin1 x=prin2 / markerchar=alcohol;
run;
/* 3b */
proc cluster data=outs2 method=average ccc pseudo print=15;
  var prin1 prin2;
  copy alcohol;
  ods select Dendrogram CccPsfAndPsTSqPlot;
run;
/* 3c */
proc tree noprint ncl=3 out=outs3;
   copy alcohol prin1 prin2;
run;
proc sort data=outs3;
 by cluster;
run;
proc freq data=outs3;
  tables cluster*alcohol/ nopercent norow nocol;
run;
/* 4a */
proc stepdisc data=wine sle=.05 sls=.05;
   	class alcohol;
   	var malic_acid--proline;
	ods select Summary;
run;
proc discrim data=wine pool=test crossvalidate manova;
  	class alcohol;
  	var malic_acid--proline;
	ods select ChiSq MultStat ClassifiedCrossVal ErrorCrossVal;
run;
