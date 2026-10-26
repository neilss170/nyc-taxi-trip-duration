// Trip-duration model in the browser: the same features as src/tripdur/features.py, then the exported
// LightGBM trees. tests/test_parity.py runs this file under Node and checks it against Python.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.TripModel = factory();
})(typeof self !== "undefined" ? self : this, function () {
  const DEG = Math.PI / 180;
  const sq = (v) => v * v;

  function haversineKm(p, lat1, lon1, lat2, lon2) {
    const p1 = lat1 * DEG, p2 = lat2 * DEG, dp = p2 - p1, dl = (lon2 - lon1) * DEG;
    const a = sq(Math.sin(dp / 2)) + Math.cos(p1) * Math.cos(p2) * sq(Math.sin(dl / 2));
    return 2 * p.earth_km * Math.asin(Math.sqrt(a));
  }
  const project = (p, lat, lon) => [(lon - p.lon0) * p.km_per_deg_lon, (lat - p.lat0) * p.km_per_deg_lat];
  const toGrid = (p, x, y) => [x * p.grid_sin + y * p.grid_cos, x * p.grid_cos - y * p.grid_sin];

  function nearestAirport(p, lat, lon) {
    for (let i = 0; i < p.airports.length; i++) {
      if (haversineKm(p, lat, lon, p.airports[i][0], p.airports[i][1]) < p.airport_km) return i + 1;
    }
    return 0;
  }
  function cluster(p, lat, lon) {
    const [x, y] = project(p, lat, lon);
    let best = 0, bestD = Infinity;
    p.centroids.forEach((c, i) => { const d = sq(x - c[0]) + sq(y - c[1]); if (d < bestD) { bestD = d; best = i; } });
    return best;
  }

  // trip: {pickup_datetime: "YYYY-MM-DD HH:MM[:SS]" (New York local time), pickup_latitude, pickup_longitude,
  //        dropoff_latitude, dropoff_longitude, passenger_count, vendor_id}
  function features(model, trip) {
    const p = model.params;
    const [date, time] = trip.pickup_datetime.split(" ");
    const [y, m, d] = date.split("-").map(Number), [hh, mm] = time.split(":").map(Number);
    const day = Date.UTC(y, m - 1, d);
    const weekday = (new Date(day).getUTCDay() + 6) % 7;
    const plat = trip.pickup_latitude, plon = trip.pickup_longitude, dlat = trip.dropoff_latitude, dlon = trip.dropoff_longitude;
    const [px, py] = project(p, plat, plon), [dx, dy] = project(p, dlat, dlon);
    const [pa, ps] = toGrid(p, px, py), [da, ds] = toGrid(p, dx, dy);
    const pc = cluster(p, plat, plon), dc = cluster(p, dlat, dlon);
    const f = {
      vendor_id: trip.vendor_id, passenger_count: trip.passenger_count,
      pickup_latitude: plat, pickup_longitude: plon, dropoff_latitude: dlat, dropoff_longitude: dlon,
      pickup_avenue_km: pa, pickup_street_km: ps, dropoff_avenue_km: da, dropoff_street_km: ds,
      haversine_km: haversineKm(p, plat, plon, dlat, dlon),
      grid_km: Math.abs(da - pa) + Math.abs(ds - ps),
      avenue_km: da - pa, street_km: ds - ps,
      bearing_deg: Math.atan2(dx - px, dy - py) * (180 / Math.PI),
      hour: hh + mm / 60, weekday, day_of_year: (day - Date.UTC(y, 0, 1)) / 86400000 + 1,
      is_weekend: weekday >= 5 ? 1 : 0, is_holiday: p.holidays.includes(date) ? 1 : 0,
      pickup_airport: nearestAirport(p, plat, plon), dropoff_airport: nearestAirport(p, dlat, dlon),
      te_pickup: p.tables.te_pickup[pc], te_dropoff: p.tables.te_dropoff[dc], te_pair: p.tables.te_pair[pc * p.n_clusters + dc],
    };
    return model.feature_names.map((name) => f[name]);
  }

  // sum of tree outputs = predicted log1p(seconds)
  function predictLog(model, x) {
    let total = 0;
    for (const t of model.trees) {
      if (!t.f.length) { total += t.v[0]; continue; }
      let node = 0;
      while (node >= 0) node = x[t.f[node]] <= t.t[node] ? t.l[node] : t.r[node];
      total += t.v[-node - 1];
    }
    return total;
  }

  function predictMinutes(model, trip) { return Math.expm1(predictLog(model, features(model, trip))) / 60; }

  // pickup and dropoff location clusters, for looking up similar past trips
  const clusters = (model, trip) => [cluster(model.params, trip.pickup_latitude, trip.pickup_longitude),
    cluster(model.params, trip.dropoff_latitude, trip.dropoff_longitude)];

  return { features, predictLog, predictMinutes, clusters, haversineKm: (model, a, b, c, d) => haversineKm(model.params, a, b, c, d) };
});
