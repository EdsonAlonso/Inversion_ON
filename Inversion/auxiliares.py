import numpy as np
from random import seed

def generate_far_from_center( n ):
    a = np.arange( 0, 10*n, 10 )
    center = np.mean( a )
    
    if n%2 == 0:
        N = int( n/2 )
        half_before = np.arange( 0, 5*N, 5 )
        half_after = np.arange( 15*N, 20*N, 5 )
        return np.r_[ half_before, half_after ]
    
    N = int( n/2 )
    half_before = np.arange( 0, 5*N + 1, 5 )
    half_after = np.arange( 15*N, 20*N, 5 )
    return np.r_[ half_before, half_after ]


def generate_near_from_center( n, percentage_away = 0.1 ):
    assert percentage_away <= 1, 'Percentage must be between 0 and 1'
    a = np.arange( 0, 10*n, 10 )
    center = np.mean( a )
    side_percentage = percentage_away*center
    return np.linspace( center - side_percentage, center + side_percentage, n )


def add_gaussian_noise( array, mean, std, Seed = 69 ):
    seed(Seed)
    gn = np.random.normal( mean, std, size = np.shape( array ) )
    return array + gn

def generate_L_random_sequences( std,N, L, Seed = 10 ):
    """
    std: standard deviation
    N: int: number of observations
    L: int: number of random arrays to be generated
    """
    seed(Seed)
    e = np.random.normal( 0, std, (L,N) )
    return e


def a_estimator( x, y ):
    assert len( x ) == len( y ), 'X and Y must have the same lenght'
    x_mean = np.mean( x )
    N = len( x )
    numerator = 0
    denominator = 0
    for i in range( N ):
        numerator += y[ i ]*( x[ i ] - x_mean )
        denominator += ( x[ i ] - x_mean )**2
    return numerator/denominator


def b_estimator( x, y, a):
    assert len( x ) == len( y ), 'X and Y must have the same lenght'
    N = len( x )
    numerator1 = np.sum( y )
    numerator2 = a*np.sum( x )
    numerator = numerator1 - numerator2
    return numerator / N

def variance_a( x, error_variance ):
    N = len( x )
    x_mean = np.mean( x )
    denominator = 0
    for i in range( N ):
        denominator += ( x[ i ] - x_mean )**2
        
    return error_variance/denominator


def variance_b( x, error_variance ):
    N = len( x )
    x_mean = np.mean( x )
    numerator = 0
    denominator = 0
    for i in range( N ):
        numerator += x[ i ]**2
        denominator += ( x[ i ] - x_mean )**2
        
    return ( error_variance*numerator )/( N*denominator )

def data_missfit( ytrue, ycalc ):
    assert len( ytrue ) == len( ycalc ), 'Both arrays must have the same lenght'
    N = len( ytrue )
    missfit = 0
    for i in range( N ):
        missfit += ( ytrue[ i ] - ycalc[ i ] )**2
        
    return missfit

def std_noised_error( ytrue, ycalc, M ):
    assert len( ytrue ) == len( ycalc ), 'Both arrays must have the same lenght'
    N = len( ytrue )
    numerator = data_missfit( ytrue, ycalc )
    denominator = N-M
    
    return np.sqrt( numerator/denominator )

def std_parameter_error( parameters ):
    if not isinstance( parameters, np.ndarray ):
        parameters = np.array( parameters )
    param_mean = np.mean( parameters, axis = 1 )
    L = len( parameters[ 0 ] )
    s = [ ]
    numerator = 0
    for i in range( L ):
        numerator += ( parameters[:,i] - param_mean )**2
    
    denominator = L - 1
        
    s = (numerator/denominator )**(0.5)
    return s
    















