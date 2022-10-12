import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import axios from 'axios';
import { useState } from 'react';
import { useEffect } from 'react';
import Icon from '../components/icon';
import ContextIcon from '/public/context.png';

export default (props) => {
  const [datasets, setDatasets] = useState([]);

  useEffect(() => {
    axios
      .get('http://localhost:8081/datasets')
      .then((response) => {
        const { data } = response;
        setDatasets(data);
      })
      .catch((err) => {
        console.log(err);
      });
  }, []);

  const switchDs = (id) => () => {
    axios.post(`http://localhost:8081/switch/${id}`);
  };

  return (
    <div className="folder">
      <Tabs>
        <TabList>
          {datasets.map((dataset) => (
            <Tab key={dataset}>{dataset}</Tab>
          ))}
        </TabList>
        {datasets.map((ds, index) => (
          <TabPanel key={ds}>
            <button className="button" onClick={switchDs(index)}>
              Activate
            </button>
            <div className="image-container"></div>
          </TabPanel>
        ))}
      </Tabs>
    </div>
  );
};
