
const fs = require('fs');
const Papa = require('papaparse');
const path = require('path');

const strategyPath = path.join(process.cwd(), 'public/data/strategy_daily_pnl.csv');
const factorsPath = path.join(process.cwd(), 'public/data/individual_factors_daily_pnl.csv');

console.log("Testing Strategy CSV Parsing...");
const strategyCsv = fs.readFileSync(strategyPath, 'utf8');
Papa.parse(strategyCsv, {
    header: true,
    dynamicTyping: true,
    complete: (results) => {
        const data = results.data;
        console.log("First 2 strategy rows:", data.slice(0, 2));

        let cumulative = 0;
        const processed = data.map((d) => {
            if (typeof d.daily_ret === 'number') {
                cumulative += d.daily_ret;
            }
            return { ...d, cumulative_ret: cumulative };
        }).filter(d => typeof d.daily_ret === 'number');

        console.log("First 2 processed:", processed.slice(0, 2));
        console.log("Any NaNs?", processed.some(d => isNaN(d.cumulative_ret)));
    }
});

console.log("\nTesting Factors CSV Parsing...");
const factorsCsv = fs.readFileSync(factorsPath, 'utf8');
Papa.parse(factorsCsv, {
    header: true,
    dynamicTyping: true,
    complete: (results) => {
        const data = results.data;
        // Filter out potential empty last lines which PapaParse often includes
        const validData = data.filter(d => Object.keys(d).length > 1);
        console.log("First factor row keys:", Object.keys(validData[0]));
        console.log("First factor row values:", validData[0]);
    }
})
