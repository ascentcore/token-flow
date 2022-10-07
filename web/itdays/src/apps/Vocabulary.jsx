import axios from 'axios';
import { useState } from 'react';
import { useEffect } from 'react';
import * as d3 from 'd3';
import { useRef } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';

export default (props) => {
  const { state } = props;
  const [data, setData] = useState([]);

  console.log('Props', props)

  useEffect(() => {
    console.log('calll')
    axios.get(`http://localhost:8081/vocabulary`).then((response) => {
      console.log(response.data);
      setData(response.data);
    });
  }, [state]);

  return (
    <div className="vocabulary-body">
      <table cellSpacing={0}>
        <thead>
          <tr>
            <th>Token</th>
          </tr>
        </thead>
        <tbody>
          {data.map((token) => (
            <tr key={token}>
              <td>{token}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
