/* Note: I used SAS studio to obtain the results for my report file. */
data sleep;
    infile 'C:\Stat 448\sleep.csv' dlm=',';
	*infile '/home/clever_g0/Stat 448/sleep.csv' dlm=',';
	input species $ bodyweight brainweight nondreamingsleep dreamingsleep
		totalsleep maxlifespan gestationtime predationindex sleepexposureindex
		overalldangerindex;
	maxlife10=maxlifespan < 10;
	drop nondreamingsleep dreamingsleep;
run;
/* 1a */
proc logistic data = sleep plots=influence;
	class maxlife10 predationindex sleepexposureindex overalldangerindex;
	model maxlife10 = bodyweight--totalsleep gestationtime--overalldangerindex / selection = stepwise;
	output out=diagnostics1 cbar=cbar1;
	ods select ModelBuildingSummary InfluencePlots;
run;
/* 1bc */
proc logistic data = sleep;
	class maxlife10;
	model maxlife10 = gestationtime / lackfit;
	ods select FitStatistics GlobalTests LackFitChiSq ParameterEstimates OddsRatios;
run;
/* 2a */
proc logistic order=data data = sleep plots=influence;
    class maxlife10;
	model maxlife10 = bodyweight--totalsleep gestationtime--overalldangerindex / selection = stepwise;
	output out=diagnostics2 cbar=cbar2;
	ods select ModelBuildingSummary InfluencePlots;
run;
proc logistic order=data data = diagnostics2 plots=influence;
	class maxlife10;
	where cbar2<1;
	model maxlife10 = bodyweight--totalsleep gestationtime--overalldangerindex / selection = stepwise;
	ods select ModelBuildingSummary InfluencePlots;
run;
/* 2bc */
proc logistic order=data data = diagnostics2 plots=influence;
	class maxlife10;
	where cbar2<1;
	model maxlife10 = predationindex sleepexposureindex / lackfit;
	ods select FitStatistics GlobalTests LackFitChiSq ParameterEstimates OddsRatios InfluencePlots;
run;
/* 3 */
proc genmod data=sleep plots=(stdreschi stdresdev cooksd);
    class sleepexposureindex predationindex;
	model gestationtime = sleepexposureindex brainweight bodyweight predationindex/ dist=poisson link=log 
				scale=d type1 type3;
	output out = poisres1 pred = predbp1 stdreschi = schires1 cooksd = cd1;
	ods select ModelInfo ParameterEstimates Type1 Type3 DiagnosticPlot;
run;
proc print data=poisres1;
    where cd1>1;
    *where abs(schires3)>2;
run;
proc genmod data=poisres1 plots=(stdreschi stdresdev cooksd);
    class sleepexposureindex predationindex;
    where cd1<1;
	model gestationtime = sleepexposureindex brainweight bodyweight predationindex/ dist=poisson link=log 
				scale=d type1 type3;
	output out = poisres2 pred = predbp2 stdreschi = schires2 cooksd = cd2;
	ods select ModelInfo ParameterEstimates Type1 Type3 DiagnosticPlot;
run;
proc print data=poisres2;
    where cd2>1;
    *where abs(schires3)>2;
run;
proc genmod data=poisres2 plots=(stdreschi stdresdev cooksd);
    class sleepexposureindex predationindex;
    where cd2<1;
	model gestationtime = sleepexposureindex brainweight bodyweight predationindex/ dist=poisson link=log 
				scale=d type1 type3;
	output out = poisres3 pred = predbp3 stdreschi = schires3 cooksd = cd3;
	ods select ModelInfo ParameterEstimates Type1 Type3 DiagnosticPlot;
run;
proc print data=poisres3;
    where cd3>1;
    *where abs(schires3)>2;
run;
proc genmod data=poisres2 plots=(stdreschi stdresdev cooksd);
    class sleepexposureindex;
    where cd2<1;
	model gestationtime = brainweight sleepexposureindex/ dist=poisson link=log 
				scale=d type1 type3;
	output out = poisres4 pred = predbp4 stdreschi = schires4 cooksd = cd4;
	ods select ModelInfo ParameterEstimates Type1 Type3 DiagnosticPlot;
run;
proc print data=poisres4;
    where cd4>1;
    *where abs(schires3)>2;
run;
