/* Note: I used SAS studio to obtain the results for my report file. */
data pottery;
	infile "C:\\Stat 448\pottery.dat" expandtabs;
	*infile '/home/clever_g0/Stat 448/pottery.dat' expandtabs;
	input id Kiln Al Fe Mg Ca Na K Ti Mn Ba;
	drop id;
run;
/* 1ab */
proc princomp data=pottery;
   id Kiln;
   ods exclude SimpleStatistics;
run;
/* 1c */
proc princomp data=pottery plots= score(ellipse ncomp=3);
   id Kiln;
   ods select ScorePlot;
run;
/* 2ab */
proc princomp data=pottery cov;
   id Kiln;
   ods exclude SimpleStatistics;
run;
/* 2c */
proc princomp data=pottery cov plots= score(ellipse ncomp=2);
   id Kiln;
   ods select ScorePlot;
run;
/* 3a */
proc cluster data=pottery method=average ccc pseudo print=15;
  var Al--Ba;
  copy Kiln;
  ods select Dendrogram CccPsfAndPsTSqPlot;
run;
proc tree noprint ncl=3 out=outs;
   copy Kiln Al--Ba;
run;
proc sort data=outs;
 by cluster;
run;
proc print data=outs;
run;
* do means analysis on variables by cluster;
proc means data=outs;
 var Al--Ba;
 by cluster;
run;
/* 3b */
proc freq data=outs;
  tables cluster*Kiln/ nopercent norow nocol;
run;
/* 4a */
proc cluster data=pottery method=average ccc std pseudo print=15;
  var Al--Ba;
  copy Kiln;
  ods select Dendrogram CccPsfAndPsTSqPlot;
run;
proc tree noprint ncl=3 out=outss;
   copy Kiln Al--Ba;
run;
proc sort data=outss;
 by cluster;
run;
proc print data=outss;
run;
* do means analysis on variables by cluster;
proc means data=outss;
 var Al--Ba;
 by cluster;
run;
/* 4b */
proc freq data=outss;
  tables cluster*Kiln/ nopercent norow nocol;
run;

