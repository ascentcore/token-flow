import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import axios from 'axios';
import { useState } from 'react';
import { useEffect } from 'react';
import { triggerEvent } from '../events';
import Baby from '/public/baby.png';
import MechanicalVsSoftware from '/public/mechanicalvssoftware.png';
import Politician from '/public/politician.png';
import Chatbots from '/public/chatbots.png';

const images = [
  Baby,
  MechanicalVsSoftware,
  Politician,  
  Chatbots,
];

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
    axios.post(`http://localhost:8081/switch/${id}`).then((data) => {
      triggerEvent('dataset', id);
    });
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
            <div className="image-container">
              <img src={images[index]} />
            </div>

            <button className="button full" onClick={switchDs(index)}>
              Activate
            </button>
          </TabPanel>
        ))}
      </Tabs>
    </div>
  );
};
