// Deprecated library usage
const crypto = require('crypto');
const fs = require('fs');

// Global variables
var globalVar1 = "This is global";
var globalVar2 = 42;

// Function with too many parameters
function processData(param1, param2, param3, param4, param5, param6) {
    console.log("Processing data...");
}

// Function lacking comments
function performOperation(a, b) {
    if (a > 10) {
        return a + b;
    } else {
        return a - b;
    }
}

// Function with excessive nesting
function nestedFunction(x) {
    if (x > 0) {
        if (x < 100) {
            if (x % 2 === 0) {
                for (let i = 0; i < x; i++) {
                    console.log(i);
                }
            }
        }
    }
}

// Hardcoded secret
const apiKey = "12345-ABCDE";

// SQL injection vulnerability
function authenticateUser(username, password) {
    const query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'";
    console.log(query);
}

// Weak cryptography
const hash = crypto.createHash('md5').update('password').digest('hex');

// File handling without specifying encoding
fs.readFile('data.txt', (err, data) => {
    if (err) throw err;
    console.log(data);
});

// Exception without a message
throw new Error();

// Long chained calls
const result = someObject.method1().method2().method3().method4();

// Unreachable code
function calculateSum(a, b) {
    return a + b;
    console.log("This code is unreachable");
}

// Non-camelCase variable name
const my_variable = 10;

// Shadowing built-in name
var eval = "This shadows a built-in";