/* Note: I used SAS studio to obtain the results for my report file. */
/* data for exercises 1 and 2 */
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
/* Excercise 1 */
proc freq order=data data=haireyes;
	tables eyecolor*haircolor/ norow nocol nopercent expected chisq;
	weight count;
run;
/* Excercise 2 */
data haireyes2;
	input eyecolor $ haircolor $ count;
	datalines;
	light fair 688
	light dark 188
	dark fair 98
	dark dark 681
	;
proc freq order=data data=haireyes2;
	tables eyecolor*haircolor/ nocol nopercent expected chisq riskdiff;
	weight count;
	ods select CrossTabFreqs ChiSq RiskDiffCol1 FishersExact;
run;
/* data for exercises 3 and 4 */
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
/* Excercise 3 */
proc freq order=data data=heartbpchol;
	tables BP_Status*Chol_Status/ norow nocol nopercent expected chisq;
run;
/* Excercise 4 */
proc anova data=heartbpchol;
	class BP_Status;
	model Cholesterol=BP_Status; 
	means BP_Status/ hovtest tukey cldiff;
	ods select OverallAnova FitStatistics HOVFTest CLDiffs;
run;
