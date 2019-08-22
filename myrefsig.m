function y = myrefsig(f, H, w_len, Fs)

num_pts = w_len * Fs;
j = 1 : num_pts ;
t = j / Fs ; 
i = 1 : H ;
i_2 = 2*i;
i_2_ = i_2 - 1;

fun_sin = @(a,b) sin(2*pi*(a*f)*b);
fun_cos = @(a,b) cos(2*pi*(a*f)*b);

y_sin = bsxfun(fun_sin,i',t);
y_cos = bsxfun(fun_cos,i',t);

y(i_2_',:) = y_sin;
y(i_2',:) = y_cos;

end 
