ods rtf file="C:\Stat 448\HW3_Solution_Spring2018.rtf";
/* Read dataset */
data cardata;
	set sashelp.cars;
	keep cylinders origin type mpg_highway;
	where type not in('Hybrid','Truck','Wagon','SUV') and 
		cylinders in(4,6) and origin ne 'Europe';
run;


data housing;
	infile 'C:\Stat 448\housing.data';
	input crim zn indus chas nox rm age dis rad tax ptratio b lstat medv;
	logmedv = log(medv);
	over25kSqFt = 'none';
	if zn > 0 then over25kSqFt = 'some';
	taxlevel = 'higher';
	if tax < 500 then taxlevel = 'lower';
	ptlevel = 'medium';
	if ptratio < 15 then ptlevel = 'lower';
	if ptratio > 20 then ptlevel = 'higher';
	drop zn tax ptratio b lstat rad dis chas;
run;
data housing;
	set housing;
	where medv<50;
	drop medv;
run;


/* Exercise 1 */
proc tabulate data=cardata;
	class cylinders origin type;
	var mpg_highway;
	table cylinders*origin*type, mpg_highway*(mean std n);
run;
proc glm data=cardata;
	class cylinders origin type;
	model mpg_highway=cylinders origin type;
	ods select ModelAnova;
run;
proc glm data=cardata;
	class cylinders origin type;
	model mpg_highway=cylinders type;
	ods select OverallAnova ModelAnova FitStatistics;
run;
proc glm data=cardata;
	class cylinders origin type;
	model mpg_highway=cylinders|type;
	lsmeans cylinders type cylinders*type/ tdiff=all pdiff cl;
	ods select OverallAnova ModelAnova FitStatistics LSMeans 
		LSMeanDiffCL;
run;



* Exercise 2;
proc reg data=housing;
	model logmedv = age;
	where crim<1;
	output out=diag2a CookD=cd;
	ods select DiagnosticsPanel;
run;
proc reg data=diag2a;
	model logmedv = age;
	where crim<1 and cd <.06;
	ods select ANOVA ParameterEstimates FitStatistics DiagnosticsPanel;
run;

* Exercise 3;

proc reg data=housing;
	model logmedv = age indus nox rm / vif;
	where crim<1;
    output out=diag3a CookD=cd;
	ods select ANOVA ParameterEstimates FitStatistics DiagnosticsPanel;
run;


* Exercise 4;
proc reg data=housing;
	model logmedv = age indus nox rm 
		/selection=stepwise sle=.05 sls=.05;
	where crim<1;
	output out=diag4a CookD=cd;
	ods select SelectionSummary DiagnosticsPanel;
run;
proc reg data=diag4a;
	model logmedv = age indus nox rm;
	where crim<1 and cd<.1;
	output out=diag4a2 CookD=cd2;
	ods select DiagnosticsPanel;
run;
proc reg data=diag4a2;
	model logmedv = age indus nox rm;
	where crim<1 and cd2<.06;
	ods select ANOVA ParameterEstimates FitStatistics 
		DiagnosticsPanel ResidualPlot;
run;

ods rtf close;
