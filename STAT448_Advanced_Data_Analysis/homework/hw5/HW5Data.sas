data pottery;
	infile "C:\\Stat 448\pottery.dat" expandtabs;
	input id Kiln Al Fe Mg Ca Na K Ti Mn Ba;
	drop id;
run;
