data sleep;
	infile 'C:\Stat 448\sleep.csv' dlm=',';
	input species $ bodyweight brainweight nondreamingsleep dreamingsleep
		totalsleep maxlifespan gestationtime predationindex sleepexposureindex
		overalldangerindex;
	maxlife10=maxlifespan < 10;
	drop nondreamingsleep dreamingsleep;
run;


