/* Note: I used SAS studio to obtain the results for my report file. */
/* data for Exercise 1 */
data cardata;
	set sashelp.cars;
	keep cylinders origin type mpg_highway;
	where type not in('Hybrid','Truck','Wagon','SUV') and 
		cylinders in(4,6) and origin ne 'Europe';
run;
/* 1(a) cross-tabulation */
proc tabulate data=cardata;
	class Cylinders Origin Type;
	var MPG_Highway;
	table Cylinders*Origin*Type, MPG_Highway*(mean std n);
run;
/* 1(b) three-way main effects model */
proc glm data=cardata;
  class Cylinders Origin Type;
  model MPG_Highway = Cylinders Origin Type;
  ods select OverallANOVA ModelANOVA FitStatistics;
run;
proc glm data=cardata;
  class Cylinders Type;
  model MPG_Highway = Cylinders Type;
  ods select OverallANOVA ModelANOVA FitStatistics;
run;
/* 1(c) add interactions and find differences */
proc glm data=cardata;
	class Cylinders Type;
	model MPG_Highway = Cylinders|Type;
	lsmeans Cylinders|Type / pdiff=all cl;
	ods select ModelANOVA OverallANOVA FitStatistics LSMeans LSMeanDiffCL;;
run;
/* data for Exercise 2,3 and 4 */
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
/* 2(a, b) */
data housing;
	set housing;
	where crim<1;
run;
proc reg data=housing;
    model logmedv = age;
    ods select ANOVA FitStatistics ParameterEstimates DiagnosticsPanel;
    output out=diagnostics cookd= cd;
run;
proc print data=diagnostics;
    where cd > 16/325;
run;
proc reg data=diagnostics;
	model logmedv = age;
	where cd <= 16/325;
	ods select ANOVA FitStatistics ParameterEstimates DiagnosticsPanel;
run;
/* 3(a) */
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
data housing;
	set housing;
	where crim<1;
run;
proc reg data=housing; 
    model logmedv = age indus nox rm / vif;
    ods select ANOVA FitStatistics ParameterEstimates DiagnosticsPanel;
run;
/* 3(b) */
proc sgscatter data=housing;
    matrix age indus nox rm;
run;
/* 4(a, b) */
data housing;
	* infile '/home/clever_g0/Stat 448/housing.data';
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
data housing;
	set housing;
	where crim<1;
run;
proc reg data=housing;
    model logmedv = indus--age/ selection=backward sls=.05;
    ods exclude ResidualPlot;
    output out=diagnostics3 cookd=cd3;
run;
proc reg data=diagnostics3;
	model logmedv = age indus nox rm;
	where cd3 <= 16/325;
	ods select ANOVA FitStatistics ParameterEstimates DiagnosticsPanel;
run;