// Runs site/predict.js on the parity fixture and prints its features and predictions as JSON.
const fs = require("fs");
const path = require("path");
const TripModel = require("../site/predict.js");

const root = path.join(__dirname, "..");
const model = JSON.parse(fs.readFileSync(path.join(root, "site", "model", "model.json"), "utf8"));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", "parity.json"), "utf8"));
const features = fx.rows.map((r) => TripModel.features(model, r));
const prediction_log = features.map((x) => TripModel.predictLog(model, x));
process.stdout.write(JSON.stringify({ features, prediction_log }));
