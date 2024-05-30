lc1 = 0.05;   // 边界上较小的特征长度
lc2 = 0.1;   // 内部较大的特征长度

lx = 5;
ly = 5;
r = 1;

// 定义矩形的四个角和圆弧的点
Point(1) = {0, 0, 0, lc1};    
Point(2) = {r, 0, 0, lc1};
Point(3) = {0, r, 0, lc1};
Point(4) = {-r, 0, 0, lc1};
Point(5) = {-lx, 0, 0, lc2};   
Point(6) = {-lx, ly, 0, lc2}; 
Point(7) = {lx, ly, 0, lc2};  
Point(8) = {lx, 0, 0, lc2};  

// 定义圆弧和矩形的边
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};  
Line(3) = {4, 5}; 
Line(4) = {5, 6};  
Line(5) = {6, 7};
Line(6) = {7, 8};
Line(7) = {8, 2};

// 创建一个曲线环并定义一个平面表面
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Plane Surface(1) = {1};

// 物理特性
Physical Line(1) = {1, 2, 3, 4, 5, 6, 7};  // 边界
Physical Surface(1) = {1};           // 区域

// 生成网格
Mesh 2;
