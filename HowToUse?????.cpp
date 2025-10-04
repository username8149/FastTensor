#include "Ftensor.hpp"
#include <iostream>
/* How to use The Tensor core. 
if not work -> ¯\_(ツ)_/¯  */

using namespace std;

int main() {
    //create Tensor
    auto x = Tensor<double>::random(-1.0, 1.0, {20});
    auto y = Tensor<double>::random(-1.0, 1.0, {10,2});
    
    // Operation 
    auto e = (x + y.reshape({20})).evaluate();
    
    //print
    
    cout << "x: \n"; x.print();
    cout << "y: \n"; y.print();
    cout << "x + y: "; e.print();
}

/*why you make this

EXACTLY i Made this Because i am iron man ;)

*/
