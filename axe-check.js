// const { chromium } = require('playwright');  // Import Playwright
// const axe = require('axe-core');              // Import axe-core
// const { stdin, stdout } = require('process'); // Import stdin and stdout for handling input/output
//
// // Initialize a variable to store the HTML content passed through stdin
// let htmlContent = '';
//
// // Collect data from stdin (this is where the HTML content will come in)
// stdin.on('data', chunk => {
//     htmlContent += chunk;
// });
//
// stdin.on('end', async () => {
//     try {
//         // Launch a browser instance using Playwright
//         const browser = await chromium.launch();
//
//         // Create a new browser page
//         const page = await browser.newPage();
//
//         // Set the page content to the HTML passed via stdin
//         await page.setContent(htmlContent);
//
//         // Inject axe-core into the page
//         await page.addScriptTag({ content: axe.source });
//
//         // Run axe-core to check for accessibility issues
//         const results = await page.evaluate(() => axe.run());
//
//         // Calculate the score: percentage of passed criteria
//         const totalRules = results.violations.length + results.passes.length + results.incomplete.length;
//         const passedRules = results.passes.length;
//
//         const score = totalRules > 0 ? (passedRules / totalRules) * 100 : 100;
//
//         // Add the score to the results
//         results.score = score;
//
//         // Output the results in JSON format
//         stdout.write(JSON.stringify(results, null, 2));
//
//         // Close the browser after the check
//         await browser.close();
//     } catch (error) {
//         // If there's an error, log it and exit the process with failure
//         console.error("Error:", error.message);
//         process.exit(1);
//     }
// });
//


const { chromium } = require('playwright');  // Headless browser
const axe = require('axe-core');             // Accessibility engine
const { stdin, stdout } = require('process');

// Get command-line arguments
const args = process.argv.slice(2); // Excludes 'node' and script name
const urlArg = args.find(arg => arg.startsWith("http"));

if (urlArg) {
    // ✅ Mode 1: Analyze a URL directly
    (async () => {
        try {
            const browser = await chromium.launch();
            const page = await browser.newPage();

            await page.goto(urlArg, { waitUntil: 'networkidle' });
            await page.addScriptTag({ content: axe.source });

            const results = await page.evaluate(() => axe.run());
            const total = results.violations.length + results.passes.length + results.incomplete.length;
            const score = total > 0 ? (results.passes.length / total) * 100 : 100;

            results.score = score;
            stdout.write(JSON.stringify(results, null, 2));

            await browser.close();
        } catch (error) {
            console.error("URL Check Error:", error.message);
            process.exit(1);
        }
    })();
} else {
    // ✅ Mode 2: Analyze HTML from stdin
    let htmlContent = '';
    stdin.on('data', chunk => htmlContent += chunk);
    stdin.on('end', async () => {
        try {
            const browser = await chromium.launch();
            const page = await browser.newPage();

            await page.setContent(htmlContent);
            await page.addScriptTag({ content: axe.source });

            const results = await page.evaluate(() => axe.run());
            const total = results.violations.length + results.passes.length + results.incomplete.length;
            const score = total > 0 ? (results.passes.length / total) * 100 : 100;

            results.score = score;
            stdout.write(JSON.stringify(results, null, 2));

            await browser.close();
        } catch (error) {
            console.error("HTML Check Error:", error.message);
            process.exit(1);
        }
    });
}
