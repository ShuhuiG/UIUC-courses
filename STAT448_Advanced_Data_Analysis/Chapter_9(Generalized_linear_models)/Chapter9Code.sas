data uscrime;
	infile 'c:\Stat 448\uscrime.dat' expandtabs;
    input R Age S Ed Ex0 Ex1 LF M N NW U1 U2 W X;
run;
proc glm data=uscrime;
	model R = ex0 x ed age u2/ ss1 ss3;
	ods select ParameterEstimates ModelANOVA;
run;
proc genmod data=uscrime;
	* normal and identity are defaults for dist and link when response is 
	  not given as events/trials, so there is no need to set those options 
	  here but we could set like in the textbook;
	model R = ex0 x ed age u2/ type1 type3;
	ods select ModelInfo ParameterEstimates Type1 Type3;
run;
data ghq;
  infile 'c:\Stat 448\ghq.dat' expandtabs;
  input ghq sex $ cases noncases;
  total=cases+noncases;
run;
proc logistic data=ghq;
   	class sex / param=ref;
   	model cases/total=ghq sex;
   	ods select ModelInfo ClassLevelInfo ParameterEstimates Type3;
run;
proc genmod data=ghq;
   	class sex ;
   	model cases/total=ghq sex/ type3;
   	ods select ModelInfo ParameterEstimates Type3;
run;
data ozkids;
	infile 'c:\Stat 448\ozkids.dat' dlm=' ,' expandtabs missover;
    input cell origin $ sex $ grade $ type $ days @;
        do until (days=.);
          output;
          input days @;
        end;
run;
proc print data=ozkids;
run;
* Poisson log-linear model;
proc genmod data=ozkids;
  class origin sex grade type;
  model days=origin sex grade type / dist=poisson 
		link=log type1 type3;
  ods select ModelInfo ModelFit ParameterEstimates Type1 Type3;
run;
* overdispersed Poisson log-linear model;
proc genmod data=ozkids;
  class origin sex grade type;
  model days=origin sex grade type / dist=poisson 
		link=log type1 type3 scale=deviance;
  ods select ModelInfo ModelFit ParameterEstimates Type1 Type3;
run;
* previous ANOVA model for comparison;
proc glm data=ozkids;
	class origin sex grade type;
	model days = origin sex grade type/ ss3 ss1;
	ods select OverallANOVA FitStatistics ModelANOVA;
run;
* FAP data set;
data fap;
  infile 'c:\Stat 448\fap.dat';
  input male treat base_n age resp_n;
run;
proc print data=fap;
run;
* gamma model;
proc genmod data=fap;
	model resp_n=male treat base_n age / dist=gamma 
		link=log type1 type3;	
	output out=gammares pred=presp_n stdreschi=presids
		stdresdev= dresids;
	ods select ModelInfo ModelFit ParameterEstimates Type1 Type3;	
run;
* overdispersed Poisson model;
proc genmod data=fap;
	model resp_n=male treat base_n age / dist=p 
		link=log type1 type3 scale=d;
	output out = poisres pred = presp_n stdreschi = presids
			stdresdev=dresids;	
	ods select ModelInfo ModelFit ParameterEstimates Type1 Type3;	
run;
* plot standardized Pearson and deviance residuals vs. predicted values for both models;
proc sgscatter data=gammares;
	compare y= (presids dresids) x=presp_n;
	where presp_n<100;
run;
proc sgscatter data=poisres;
	compare y= (presids dresids) x=presp_n;
	where presp_n<100;
run;
* getting plots of standardized Pearson and deviance residuals for both model
  and just displaying the model info and plots;
proc genmod data=fap plots=(stdreschi stdresdev);
	model resp_n=male treat base_n age / dist=gamma 
		link=log type1 type3;
	ods select ModelInfo DiagnosticPlot;	
run;
proc genmod data=fap plots=(stdreschi stdresdev);
	model resp_n=male treat base_n age / dist=poisson 
		link=log scale=d type1 type3;
	ods select ModelInfo DiagnosticPlot;
run;
/*
Previous was done in multiple parts to introduce different results/features in class,
we can combine code and get the plots, output data sets and additional results with 
one call to genmod for each model, as in the following: 

proc genmod data=fap plots=(stdreschi stdresdev);
	model resp_n=male treat base_n age / dist=gamma 
		link=log type1 type3;
	output out=gammares pred=presp_n stdreschi=presids	
		stdresdev=dresids;
	ods select ModelInfo ModelFit ParameterEstimates 
			Type1 Type3 DiagnosticPlot;	
run;
proc genmod data=fap plots=(stdreschi stdresdev);
	model resp_n=male treat base_n age / dist=poisson 
		link=log scale=d type1 type3;
	output out = poisres pred = presp_n stdreschi = presids
		stdresdev=dresids;
	ods select ModelInfo ModelFit ParameterEstimates 
		Type1 Type3 DiagnosticPlot;
run;

*/
