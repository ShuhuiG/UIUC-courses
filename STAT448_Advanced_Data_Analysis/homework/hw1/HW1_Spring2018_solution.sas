/* Homework 1 solution code for Stat 448, Spring 2018 at
   University of Illinois at Urbana Champaign */
/* Results and report file were generated using SAS Studio */
options nodate nonumber;
title ;
ods noproctitle;

ods text = " ";
ods text = "HW 1 Spring 2018";

ods startpage=no ;

ods graphics / reset=all height=4in width=7in;

data cars;
 set sashelp.cars;
 mpg_combo=0.55*mpg_city+0.45*mpg_highway;
 where type not eq "Hybrid";
 drop drivetrain msrp enginesize cylinders weight  ;
run;


/*-------------*/
/* Problem 1 a */
/*-------------*/
ods text = " ";
ods text = " ";
ods text="Problem 1 a";
proc sgplot data=cars;
 title "A Box Plot of the Combined MPG of All Vehicles";
 vbox mpg_combo;
run;
title;
ods text="1a) The box plot of the combined MPG has several outliers above and below the quartiles 
according to the 1.5*IQR rule. Ignoring those outliers, the mean and median are fairly close 
together (roughly 22 combined miles per gallon) and the spread of the distribution is not wide, 
rather low variability. Most vehicles get between 20 and 25 miles per gallon (combined).";


/*-------------*/
/* Problem 1 b */
/*-------------*/
ods text = " ";
ods text = " ";
ods text="Problem 1 b";
proc sgplot data=cars;
 title "A Box Plot of the Combined MPG by Vehicle Type";
 vbox mpg_combo / group=type;
run;
title;
ods text="1b) When viewing the distribution of combined MPG separated by vehicle type, we see some 
interesting things. The combined MPG of SUVs and sports cars appear to be most like normal 
distributions - seemingly symmetric, with not wide spreads. Sedans' combined MPG have lots of 
variability and the most outliers. Combined MPG for wagons also have a wide spread but may not 
be symmetric. Trucks appear to be least fuel efficient among the vehicle types. The 
distribution of combined MPG of trucks is right skewed and has at least one outlier.";


/*-------------*/
/* Problem 1 c */
/*-------------*/
ods text = " ";
ods text = " ";
ods text="Problem 1 c";
proc univariate data=cars normaltest;
 var mpg_combo ;
 id model;
 histogram mpg_combo /normal;
 probplot mpg_combo ;
 ods exclude TestsforLocation Quantiles FitQuantiles GoodnessofFit ParameterEstimates ;
run;
ods text="1c) The mean, median, and mode of MPG_Combo are similar in value which would suggest 
symmetry. However, MPG_Combo has relatively low variance (21.22) and a small IQR (4.55) with 
several outliers since the vehicles are of various type. Some extremely fuel-inefficient 
vehicles are the SUVs: Hummer H2, Ford Excursion 6.8 XLT, Mercedes-Benz G500, and 
Land Rover's Discovery SE and Range Rover HSE. Some extremely fuel-efficient vehicles 
are: three versions of the Toyota Echo, the Honda Civic, and the Volkswagen Jetta GLS TDI. 
The normality test fails and suggests that the MPG_Combo distribution is not normal.";
proc univariate data=cars normaltest;
 var invoice ;
 id model;
 histogram invoice /normal;
 probplot invoice ;
 ods exclude TestsforLocation Quantiles FitQuantiles GoodnessofFit ParameterEstimates ;
run;
ods text="The distribution of Invoice is not normal as visible from the probability plot 
and strongly skewed as visible from the histogram. The cheapest cars are Kia Rio, 
Hyundai Accent, two versions of the Toyota Echo, and the Saturn Ion. Some very expensive 
cars are all European vehicles including the Porsche 911 GT2 and four Mercedes-Benz 
vehicles - CL500, CL600, SL55 AMG, and SL 600. The mean, median, and mode are quite different 
from each other and clarifies the asymmetry of the invoice variable's distribution.";

/*-------------*/
/* Problem 1 d */
/*-------------*/
ods text = " ";
ods text = " ";
ods text="Problem 1 d";
proc sort data=cars;
 by origin;
run;
proc univariate data=cars normaltest;
 by origin;
 var mpg_combo invoice;
 histogram mpg_combo invoice /normal;
 probplot mpg_combo invoice;
 ods exclude TestsforLocation Quantiles ExtremeObs FitQuantiles GoodnessofFit 
 ParameterEstimates ;
run;
ods text="1d) There are 155 cars originating from Asia, 123 European cars, and 147 originating 
from USA in the data set. None of the distributions of MPG_Combo or Invoice by Origin are 
normal according to the tests of normality, histograms, and probability plots. The 
distribution of Invoice (in dollars), is skewed and asymmetric for each of the 3 continents. 
The respective mean, median, and mode for each of the 3 distributions are quite different from 
each other. For MPG_Combo, the American and European cars have similar means (22) and 
medians (22.15) with low standard deviations (3.63 and 4.56 respectively).";


/*-------------*/
/* Problem 2 a */
/*-------------*/
ods text = " ";
ods text = " ";
ods text="Problem 2 a";
proc univariate data=cars mu0=22000 ;
 var invoice;
 ods select TestsForLocation;
run;
ods text="2a) Since Invoice is not normal and not symmetric, we use the sign test for the null 
value of $22000. Based on the test for location table of proc univariate, we strongly reject 
the null hypothesis in favor of the alternative that the median invoice price of all vehicles 
is not $22000. "; 
*If we created the confidence interval, we would see that the population median value
 is greater than $22000.;


/*-------------*/
/* Problem 2 b */
/*-------------*/
ods text = " ";
ods text = " ";
ods text="Problem 2 b";
proc npar1way data=cars wilcoxon;
 class origin;
 var invoice;
 where origin in ('Europe','Asia');
 ods exclude KruskalWallisTest;
run;
ods text="2b) The distributions of Invoice for Europe and Asia are not normal from the normality 
tests above. Comparing the two distributions of the Invoice between European and Asian cars, 
we see the Wilcoxon rank sum test rejects the null that the two distributions are the same in 
favor of the alternative that the European cars tend to have more expensive Invoice prices. 
The box plots yield similar conclusions as the hypothesis test that the European cars have 
higher invoices.";


/*-------------*/
/* Problem 3 a */
/*-------------*/
ods text = " ";
ods text = " ";
ods text="Problem 3 a";
proc corr data=cars;
 var Invoice Horsepower Wheelbase Length mpg_combo;
 ods select PearsonCorr;
run;
ods text="3a) Length and Wheelbase have strongest correlation among the 5 variables at 0.89 
indicating as wheelbase increases the length of the vehicle tends to increase as well and 
vice versa. Invoice and Horsepower have a strong correlation of 0.82 indicating as Invoice 
price tends to increase so does the Horsepower of the car and vice versa. Horsepower and 
Combined MPG have a moderate negative correlation of -0.71 indicating as the Combined MPG 
tends to increase, the Horsepower tends to decrease and vice versa. The remaining pairwise 
correlations are much lower and indicate weaker linear relationships. The correlation 
hypothesis tests for each pair of variables are statistically significant as being unequal 
to 0 correlation. However, as aforementioned, there are only 3 pairs of moderately strong 
correlations.";

/*-------------*/
/* Problem 3 b */
/*-------------*/
ods text = " ";
ods text = " ";
ods text="Problem 3 b";
proc sort data=cars;
 by type;
run;
proc corr data=cars;
 by type;
 var Invoice Horsepower Wheelbase Length mpg_combo ;
 ods select PearsonCorr;
run;
ods text="3b) There are 60 SUVs, and 242 sedans, 49 sports cars, 24 trucks, and 30 wagons in this 
cars data set. Among SUVs, there are relatively high, positive correlations between the 
following pairs of variables: Horsepower/Invoice (0.77), Horsepower/Wheelbase (0.70), 
Horsepower/Length (0.69), and Length/Wheelbase (0.94). Among sedans, there are relatively 
high, positive correlations between the following pairs of variables: Horsepower/Invoice (0.85) 
and Length/Wheelbase (0.86). Among sports cars, there are relatively high, positive 
correlations between Horsepower and Invoice (0.80). Among trucks, there are relatively high, 
positive correlations between the following pairs of variables: Horsepower/Invoice (0.84), 
Horsepower/Wheelbase (0.75), Horsepower/Length (0.71), and Length/Wheelbase (0.94). Among 
wagons, there are relatively high, positive correlations between the following pairs of 
variables: Horsepower/Invoice (0.83) and Length/Wheelbase (0.83). These relatively high, 
positive correlations indicate as one variable tends to increases so does the other 
variable.";
ods text=" ";
ods text="Among SUVs, there are relatively high, negative correlations between the following 
pairs of variables: Horsepower/MPG_Combo (-0.76) and Invoice/MPG_Combo (-0.73). Among sedans, 
there are relatively high, negative correlations between the following pairs of variables: 
Horsepower/MPG_Combo (-0.75) and Wheelbase/MPG_Combo (-0.69). Among sports cars, there are 
relatively high, negative correlations between Horsepower and MPG_Combo (-0.84). Among trucks,  
there are relatively high, negative correlations between the following pairs of variables: 
Horsepower/MPG_Combo (-0.75), Wheelbase/MPG_Combo (-0.75), and Length/MPG_Combo (-0.74). Among 
wagons, there are relatively high, negative correlations between the following pairs of 
variables: Horsepower/MPG_Combo (-0.84) and Invoice/MPG_Combo (-0.74). These relatively high, 
negative correlations indicate as one variable tends to increase, the other variable tends to 
decrease and vice versa.";
ods text=" ";
ods text="The results here are more specific than the correlation results from part 3a). We do 
see the pairs of variables Length/Wheelbase and Invoice/Horsepower from part a) as moderately 
strong positive correlations for all 5 vehicle types. We also see the moderately strong 
negative correlation of Horsepower/Combined MPG among all vehicle types which was the case in 
part 3a). The major difference between part 3b) and 3a) is the additional pairs of variables 
achieving moderately strong correlations - positive and negative. Horsepower/Length and 
Horsepower/Wheelbase are positively correlated for several vehicle types. Invoice/Combined MPG, 
Wheelbase/Combined MPG, and Length/Combined MPG are negatively correlated for several vehicle 
types.";
