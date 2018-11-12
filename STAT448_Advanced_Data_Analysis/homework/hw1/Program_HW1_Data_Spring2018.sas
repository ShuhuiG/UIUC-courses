data cars;
 set sashelp.cars;
 where type not eq "Hybrid";
 drop drivetrain msrp enginesize cylinders weight;
run;