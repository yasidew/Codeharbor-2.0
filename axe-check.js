const { chromium } = require('playwright');  // Import Playwright
const axe = require('axe-core');              // Import axe-core
const { stdin, stdout } = require('process'); // Import stdin and stdout for handling input/output

// Initialize a variable to store the HTML content passed through stdin
let htmlContent = '';

// Collect data from stdin (this is where the HTML content will come in)
stdin.on('data', chunk => {
    htmlContent += chunk;
});

// Once the input is fully received, process it
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

stdin.on('end', async () => {
    try {
        // Launch a browser instance using Playwright
        const browser = await chromium.launch();

        // Create a new browser page
        const page = await browser.newPage();

        // Set the page content to the HTML passed via stdin
        await page.setContent(htmlContent);

        // Inject axe-core into the page
        await page.addScriptTag({ content: axe.source });

        // Run axe-core to check for accessibility issues
        const results = await page.evaluate(() => axe.run());

        // Calculate the score: percentage of passed criteria
        const totalRules = results.violations.length + results.passes.length + results.incomplete.length;
        const passedRules = results.passes.length;

        const score = totalRules > 0 ? (passedRules / totalRules) * 100 : 100;

        // Add the score to the results
        results.score = score;

        // Output the results in JSON format
        stdout.write(JSON.stringify(results, null, 2));

        // Close the browser after the check
        await browser.close();
    } catch (error) {
        // If there's an error, log it and exit the process with failure
        console.error("Error:", error.message);
        process.exit(1);
    }
});

