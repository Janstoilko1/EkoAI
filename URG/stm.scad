difference(){
    color("gray", 1.0)
    cube([100, 70, 33]);
    
    translate([1.8, 45.6, 32.6])
    color("gray", 1.0)
    cube([10.3, 6.2, 0.7]);
    
    translate([1,1,1])
    cube([98, 68, 21]);
    
    translate([39.5, 1, 1])
    cube([20.5, 25.5, 31]);
    
    translate([39.5, 43.5, 1])
    cube([20.5, 25.5, 31]);
    
    translate([1, 1, 1])
    cube([40.5, 15.5, 31]);
    
    translate([1, 52.2, 1])
    cube([40.5, 16.7, 31]);
    
    translate([12.5, 45.4, 1])
    cube([15, 7, 31]);
    
    translate([27, 39.7, 1])
    cube([13, 2, 31]);
    
    translate([27, 27.2, 1])
    cube([13, 2, 31]);
    
    translate([6, 22.5, 1])
    cube([21.5, 23, 31]);
    
    translate([14, 16, 1])
    cube([13.5, 10, 31]);
    
    translate([1, 16, 1])
    cube([13.5, 2.5, 31]);
    
    translate([1, 18, 1])
    cube([9, 27.5, 31]);
    
    translate([-0.1, 45.6, 22.5])
    color("DimGray", 1.0)
    cube([2, 6, 10.7]);
    
    translate([2.1, 51.6, 1])
    color("DimGray", 1.0)
    cube([10, 0.3, 33]);
    
    
    
    translate([-0.1, 51.59, 22.5])
    color("DimGray", 1.0)
    cube([2.3, 0.3, 11.2]);
    
    translate([-0.1, 31, 21])
    cube([2, 7, 3.7]);
    
    translate([99, 31, 21])
    cube([2, 7, 3.7]);
    
    translate([39.5, 26, 1])
    cube([3, 18, 31]);
    
    translate([59.5, 1, 1])
    cube([40, 68, 30.5]);
    
    translate([43, 27, 23.1])
    color("DarkSlateGray", 1.0)
    cube([16, 16, 11]);
    
    translate([33.5, 34.5, 23.1])
    color("DarkSlateGray", 1.0)
    cylinder(12,5,5);
    
    translate([27.9, 17, 23.1])
    color("DarkSlateGray", 1.0)
    cube([11,9.8,9]);
    union(){
        translate([38,17,23])
        cube([0.8,9,11]);
        
        translate([29,17,23])
        cube([0.8,9,11]);
        
        translate([29,25.4,23])
        cube([9.8,0.8,11]);
        
    }
    
    translate([27.9, 42, 23.1])
    color("DarkSlateGray", 1.0)
    cube([11,9.8,9]);
    union(){
        translate([38,42.5,23])
        cube([0.8,9,11]);
        
        translate([29,42.5,23])
        cube([0.8,9,11]);
        
        translate([29,42.5,23])
        cube([9.8,0.8,11]);
        
    }
    
    
    color("DarkSlateGray", 1.0)
    translate([12, 20.5, 23.5])
    cylinder(12,1.5,1.5, $fn = 50);
}