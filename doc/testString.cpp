//
// Created by hu on 2019/6/11.
//
#include "string"
#include "iostream"

using namespace std;

int main(void) {
    string a = "../data/zjudataset_0609_01//rgb/1560067205.968.png";
//    cout<<a.substr(-5,14)<<endl;
    int len = a.length();
    string x=a.substr(len - 12, 8);
    cout<<a.substr(len - 12, 8)<<endl;
    cout << stof(x)-0.004 << endl;

    /*string substr (size_t pos, size_t len) const;
    Parameters:
    pos: Position of the first character to be copied.
    len: Length of the sub-string.
    size_t: It is an unsigned integral type.
     */
}
