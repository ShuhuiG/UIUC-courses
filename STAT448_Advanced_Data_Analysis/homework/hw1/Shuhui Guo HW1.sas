/* Note: I used SAS studio to obtain the results for my report file. */
data cars;
 set sashelp.cars;
 where type not eq "Hybrid";
 drop drivetrain msrp enginesize cylinders weight;
run;
/* Excercise 1 */
/* a */
data cars;
    set cars;
    MPG_Combo = 0.55*MPG_City + 0.45*MPG_Highway;
run;
proc sgplot data=cars;
    vbox MPG_Combo;
run;
/* b */
proc sort data=cars;
    by Type;
run;
proc boxplot data=cars;
    plot MPG_Combo*Type;
run;
/* c */
proc univariate data=cars;
  var MPG_Combo Invoice;
run;
proc univariate data=cars normal;
  var MPG_Combo;
  histogram MPG_Combo /normal;
  probplot MPG_Combo;
  ods select Histogram Probplot TestsForNormality;
run;
/* d */
proc univariate data=cars normal;
  var MPG_Combo Invoice;
  histogram MPG_Combo Invoice /normal;
  probplot MPG_Combo Invoice;
  by Type; 
  ods select Moments BasicMeasures Histogram Probplot TestsForNormality;
run;
/* Excercise 2 */
/* a */
/* first check normality */
proc univariate data=cars normal;
  var Invoice;
  histogram Invoice /normal;
  probplot Invoice;
  ods select Histogram Probplot TestsForNormality;
run;
/* Invoice not normality, so not choose ttest */
proc univariate data=cars mu0=22000;
  var Invoice;
  ods select TestsForLocation;
run;
/* b */
data new_car;
set cars;
where Origin='Europe' or Origin='Asia';
run;
proc sort data=new_car;
    by Origin;
run;
proc univariate data=new_car normal;
  var Invoice;
  histogram Invoice /normal;
  probplot Invoice;
  by Origin;
  ods select Histogram Probplot TestsForNormality;
run;
proc npar1way data=new_car wilcoxon;
  class Origin;
  var Invoice;
  ods exclude KruskalWallisTest;
run;
/* Excercise 3 */
/* a */
proc corr data=cars pearson;
  var Invoice Horsepower Wheelbase Length;
  ods select PearsonCorr;
run;
/* b */
proc sort data=cars;
    by Type;
run;
proc corr data=cars pearson;
  var Invoice Horsepower Wheelbase Length;
  by Type;
  ods select PearsonCorr;
run;
