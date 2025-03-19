import React, { useState, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, PieChart, Pie, Cell,
  CartesianGrid
} from 'recharts';
import { Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import './App.css'; // For custom styling

// Parse CSV string to array of objects
const parseData = (dataString) => {
  if (!dataString) return [];
  const lines = dataString.trim().split('\n');
  const headers = lines[0].split(',');
  return lines.slice(1).map(line => {
    const values = line.split(',');
    return headers.reduce((obj, header, index) => {
      const value = values[index];
      // Convert numerical fields
      obj[header] = ['Smoking Percentage', 'Age', 'Smoking Duration', 'Cigarettes per day', 
                     'Previous Quit Attempts', 'Nicotine Dependence Score'].includes(header)
        ? parseFloat(value) || 0
        : value;
      return obj;
    }, {});
  });
};

// Main App Component
const App = ({ dataString }) => {
  const data = parseData(dataString);
  const [selectedBehavior, setSelectedBehavior] = useState('Cigarette Use (Youth)');

  // Filter data based on selected behavior
  const filteredData = useMemo(() => 
    data.filter(d => d['Smoking Behavior'] === selectedBehavior), 
    [data, selectedBehavior]
  );

  // Data processing functions
  const getStateAverages = (data) => {
    const grouped = data.reduce((acc, curr) => {
      const state = curr.Location;
      if (!acc[state]) acc[state] = { total: 0, count: 0 };
      acc[state].total += curr['Smoking Percentage'];
      acc[state].count += 1;
      return acc;
    }, {});
    return Object.keys(grouped).map(state => ({
      name: state,
      value: grouped[state].total / grouped[state].count
    }));
  };

  const getAgeVsPercentage = (data) => data.map(d => ({
    age: d.Age,
    percentage: d['Smoking Percentage']
  }));

  const getReasonsCount = (data) => {
    const counts = data.reduce((acc, curr) => {
      const reason = curr['Reason for Start Smoking'];
      acc[reason] = (acc[reason] || 0) + 1;
      return acc;
    }, {});
    return Object.keys(counts).map(key => ({ name: key, value: counts[key] }));
  };

  const getNicotineBoxData = (data) => {
    const scores = data.map(d => d['Nicotine Dependence Score']);
    const sorted = scores.sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length / 4)];
    const median = sorted[Math.floor(sorted.length / 2)];
    const q3 = sorted[Math.floor(3 * sorted.length / 4)];
    const iqr = q3 - q1;
    const min = Math.max(sorted[0], q1 - 1.5 * iqr);
    const max = Math.min(sorted[sorted.length - 1], q3 + 1.5 * iqr);
    return [{ name: 'Nicotine Dependence', min, q1, median, q3, max }];
  };

  const getQuitSuccessByState = (data) => {
    const grouped = data.reduce((acc, curr) => {
      const state = curr.Location;
      if (!acc[state]) acc[state] = { yes: 0, total: 0 };
      acc[state].total += 1;
      if (curr['Quit Success'] === 'Yes') acc[state].yes += 1;
      return acc;
    }, {});
    return Object.keys(grouped).map(state => ({
      name: state,
      value: (grouped[state].yes / grouped[state].total) * 100
    }));
  };

  const getQuitSuccessBySupport = (data) => {
    const grouped = data.reduce((acc, curr) => {
      const support = curr['Support System'];
      if (!acc[support]) acc[support] = { yes: 0, total: 0 };
      acc[support].total += 1;
      if (curr['Quit Success'] === 'Yes') acc[support].yes += 1;
      return acc;
    }, {});
    return Object.keys(grouped).map(support => ({
      name: support,
      value: (grouped[support].yes / grouped[support].total) * 100
    }));
  };

  // Prepare data
  const stateAverages = getStateAverages(filteredData);
  const ageVsPercentage = getAgeVsPercentage(filteredData);
  const reasonsCount = getReasonsCount(filteredData);
  const nicotineBoxData = getNicotineBoxData(filteredData);
  const quitSuccessByState = getQuitSuccessByState(filteredData);
  const quitSuccessBySupport = getQuitSuccessBySupport(filteredData);

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <div className="dashboard">
      <h1>Youth Smoking Behavior Dashboard</h1>
      <FormControl className="filter">
        <InputLabel>Smoking Behavior</InputLabel>
        <Select
          value={selectedBehavior}
          onChange={(e) => setSelectedBehavior(e.target.value)}
          label="Smoking Behavior"
        >
          <MenuItem value="Cigarette Use (Youth)">Cigarette Use</MenuItem>
          <MenuItem value="Smokeless Tobacco Use (Youth)">Smokeless Tobacco Use</MenuItem>
          <MenuItem value="Cessation (Youth)">Cessation</MenuItem>
        </Select>
      </FormControl>

      <div className="chart-grid">
        {selectedBehavior !== 'Cessation (Youth)' ? (
          <>
            {/* Bar Chart: Average Smoking Percentage by State */}
            <div className="chart-container">
              <h3>Average Smoking Percentage by State</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stateAverages}>
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Scatter Plot: Age vs. Smoking Percentage */}
            <div className="chart-container">
              <h3>Age vs. Smoking Percentage</h3>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <XAxis dataKey="age" name="Age" unit=" years" />
                  <YAxis dataKey="percentage" name="Percentage" unit="%" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter name={selectedBehavior} data={ageVsPercentage} fill="#82ca9d" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Bar Chart: Reasons for Starting Smoking */}
            <div className="chart-container">
              <h3>Reasons for Starting Smoking</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={reasonsCount}>
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#ff7300" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Box Plot: Nicotine Dependence Score (Simplified) */}
            <div className="chart-container">
              <h3>Nicotine Dependence Score Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={nicotineBoxData}>
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 10]} />
                  <Tooltip />
                  <Bar dataKey="min" fill="#d3d3d3" stackId="a" />
                  <Bar dataKey="q1" fill="#add8e6" stackId="a" />
                  <Bar dataKey="median" fill="#87ceeb" stackId="a" />
                  <Bar dataKey="q3" fill="#add8e6" stackId="a" />
                  <Bar dataKey="max" fill="#d3d3d3" stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        ) : (
          <>
            {/* Bar Chart: Average Attempt to Quit Percentage by State */}
            <div className="chart-container">
              <h3>Average Attempt to Quit Percentage by State</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stateAverages}>
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Bar Chart: Quit Success Rate by State */}
            <div className="chart-container">
              <h3>Quit Success Rate by State</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={quitSuccessByState}>
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis unit="%" />
                  <Tooltip />
                  <Bar dataKey="value" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Bar Chart: Quit Success Rate by Support System */}
            <div className="chart-container">
              <h3>Quit Success Rate by Support System</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={quitSuccessBySupport}>
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis unit="%" />
                  <Tooltip />
                  <Bar dataKey="value" fill="#ff7300" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Pie Chart: Reasons for Starting Smoking */}
            <div className="chart-container">
              <h3>Reasons for Starting Smoking</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={reasonsCount}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label
                  >
                    {reasonsCount.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default App;