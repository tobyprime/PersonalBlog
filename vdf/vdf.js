function cal_vdf(lambda,x,t) {
     for (var i = 0; i < t; i++) {
        x = x * x % lambda
    }
    return x;
}
