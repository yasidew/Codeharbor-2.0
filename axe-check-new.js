const { chromium } = require('playwright');  // Import Playwright
const axe = require('axe-core');              // Import axe-core
const { stdin, stdout } = require('process'); // Import stdin and stdout

let htmlContent = '';

stdin.on('data', chunk => {
    htmlContent += chunk;
});

stdin.on('end', async () => {
    try {
        const browser = await chromium.launch();
        const page = await browser.newPage();
        await page.setContent(htmlContent);
        await page.addScriptTag({ content: axe.source });

        const results = await page.evaluate(() => axe.run());

        // Count number of violations
        const violationCount = results.violations.length;

        // Output only the violation count
        stdout.write(violationCount.toString());

        await browser.close();
    } catch (error) {
        console.error("Error:", error.message);
        process.exit(1);
    }
});
