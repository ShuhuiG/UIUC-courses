
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