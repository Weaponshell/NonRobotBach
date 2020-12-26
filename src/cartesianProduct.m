function res = cartesianProduct(x)
    % Calculate the Cartesian product of a vector.
    %
    % Input: 
    % x: n x 1 vector 
    %
    % Output:
    % res: n^2 x 1 vector containing the Cartesian product of x
    
    x_len = length(x);
    res = zeros(x_len^2, 1);
    
    % Calculate Cartesian product of the values in x.
    for i = 1:x_len
        for j = 1:x_len
            idx = (i - 1) * x_len + j;
            res(idx) = x(i) * x(j);
        end
    end
end