import axios from 'axios';
import { useState } from 'react';
import { useEffect } from 'react';
import * as d3 from 'd3';
import { useRef } from 'react';
import { registerListener, unregisterListener } from '../events';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import RadialGraph from './RadialGraph';
import Graph from './Graph';
import Stimuli from './Stimuli';
import Matrix from './Matrix';
import Canvas from './Canvas';

export default (props) => {
  const { context, callBackendForText, resetStimuli, stimulate } = props;

  const containerRef = useRef(null);
  const [data, setData] = useState(null);
  const [threshold, setThreshold] = useState(0);
  const [topTokens, setTopTokens] = useState([]);
  const [svgContainer, setSvgContainer] = useState(null);
  const [sim, setSim] = useState(null);
  const [nodes, setNodes] = useState(null);
  const [links, setLinks] = useState(null);
  const [textValue, setTextValue] = useState('');
  const [state, setState] = useState(0);

  useEffect(() => {
    const callback = (data) => {
      setState(data);
    };
    registerListener('global', callback);
    registerListener(context, callback);
    return () => {
      unregisterListener(callback);
    };
  }, []);

  function keyPress(e) {
    if (e.keyCode === 13) {
      const stimulate = false;
      const sendValue = e.target.value.trim();
      setTextValue('');
      callBackendForText(context, sendValue, true);
    }
  }

  function addToContext(e) {
    setTextValue('');
    callBackendForText(context, textValue.trim(), false);
  }

  function processTopTokens(nodes) {
    nodes.sort((a, b) => b.s - a.s);

    setTopTokens(nodes.slice(0, 10));
  }

  function onClick() {
    const elems = d3.select(this).data();
    if (elems && elems.length > 0) {
      const { id } = elems[0];
      onNodeClick(id);
    }
  }

  function doNode(node) {
    node.on('click', onClick);

    node.attr('class', 'node');
    node.attr('opacity', (d) => {
      return d.s >= threshold ? 1 : 0;
    });
    node
      .append('circle')
      .attr('shape-rendering', 'crispEdges')
      .attr('class', 'stimulus')
      .attr('fill', 'rgba(255,0,0,0.2)')
      .attr('r', (d) => (Math.pow(2, d.s) - 1) * 30);
    node.append('circle').attr('r', 2).attr('shape-rendering', 'crispEdges');
    node
      .append('text')
      .text((d) => d.id)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('alignment-baseline', 'baseline')
      .attr('transform', 'translate(0, -6)');
  }

  function doLine(line) {
    line.attr('stroke', 'rgba(0,0,0,.2)').attr('stroke-width', (d) => d.weight);
  }

  function adjustOpacity() {
    if (svgContainer) {
      svgContainer
        .selectAll('.node')
        .transition()
        .duration(300)
        .attr('opacity', (d) => {
          return d.s >= threshold ? 1 : 0;
        });

      svgContainer
        .selectAll('.stimulus')
        .transition()
        .duration(300)
        .attr('r', (d) => (Math.pow(2, d.s) - 1) * 30);

      svgContainer
        .selectAll('line')
        .transition()
        .duration(300)
        .attr('stroke-opacity', (d) =>
          Math.max(d.source.s, d.target.s) - threshold > 0
            ? Math.max(d.source.s, d.target.s)
            : 0
        );
    }
  }

  useEffect(() => {
    adjustOpacity();
  }, [threshold]);

  useEffect(() => {
    axios
      .get(`http://localhost:8081/context/${props.context}/graph`)
      .then((response) => {
        const { data: graph } = response;
        processTopTokens(graph.nodes);
        setData(graph);

        const rootContainer = d3.select(containerRef.current);
        rootContainer.selectAll('*').remove();
        const svg = rootContainer.append('g');
        const width = containerRef.current.getBoundingClientRect().width;
        const height = containerRef.current.getBoundingClientRect().height;

        setSvgContainer(svg);

        const zoom = d3.zoom().on('zoom', function (event) {
          svg.attr('transform', event.transform);
        });

        rootContainer
          .call(zoom)
          .call(
            zoom.transform,
            d3.zoomIdentity.translate(width / 2, height / 2).scale(0.7)
          );

        const simulation = d3
          .forceSimulation()
          .force(
            'radial',
            d3.forceRadial(function (d) {
              return (2 - Math.pow(2, d.s)) * (width / 4);
            })
          )
          .force(
            'charge',
            d3.forceManyBody().strength((d) => -d.s * 60)
          )
          .force(
            'link',
            d3
              .forceLink()
              .id((d) => d.id)
              .strength(0.001)
          );

        var link = svg
          .append('g')
          .attr('class', 'links')
          .selectAll('line')
          .data(graph.links)
          .enter()
          .append('line');

        doLine(link);

        var node = svg
          .append('g')
          .attr('class', 'nodes')
          .selectAll('g')
          .data(graph.nodes)
          .enter()
          .append('g');

        doNode(node);

        setNodes(node);
        setLinks(link);

        const ticked = () => {
          (svgContainer || svg)
            .selectAll('line')

            .attr('x1', function (d) {
              return d.source.x;
            })
            .attr('y1', function (d) {
              return d.source.y;
            })
            .attr('x2', function (d) {
              return d.target.x;
            })
            .attr('y2', function (d) {
              return d.target.y;
            });

          //

          (svgContainer || svg).selectAll('.node').attr('transform', (d) => {
            return `translate(${d.x}, ${d.y})`;
          });
        };

        simulation.nodes(graph.nodes).on('tick', ticked);
        simulation.force('link').links(graph.links);

        setSim(simulation);
      });
  }, [context]);

  useEffect(() => {
    axios
      .get(`http://localhost:8081/context/${context}/graph`)
      .then((response) => {
        const { data: graph } = response;
        setData(graph);
      });
  }, [state]);

  return (
    <div className="context-body">
      <div className="controls">
        <input
          type="text"
          value={textValue}
          onKeyDown={keyPress}
          onChange={(e) => setTextValue(e.target.value)}
        ></input>
        <button onClick={addToContext}>Add to context</button>
        <button onClick={resetStimuli(context)}>Reset stimuli</button>
      </div>
      <div className="controls">
        Threshold
        <input
          type="range"
          value={threshold}
          min={0}
          max={1}
          step={0.01}
          onChange={(e) => {
            console.log(e.target.value);
            setThreshold(parseFloat(e.target.value));
          }}
        />
        <span style={{ width: 50 }}>{Math.floor(threshold * 100) / 100}</span>
      </div>

      <Tabs>
        <TabList>
          <Tab>RadialGraph</Tab>
          <Tab>Graph</Tab>
          <Tab>Stimuli</Tab>
          <Tab>Matrix</Tab>
          <Tab>Overview</Tab>
        </TabList>
        <TabPanel>
          <RadialGraph
            context={context}
            threshold={threshold}
            stimulate={stimulate}
          />
        </TabPanel>
        <TabPanel>
          <Graph
            context={context}
            threshold={threshold}
            stimulate={stimulate}
          />
        </TabPanel>
        <TabPanel>
          <Stimuli graph={data} context={context} stimulate={stimulate} />
        </TabPanel>
        <TabPanel>
          <Matrix graph={data} context={context} stimulate={stimulate} />
        </TabPanel>
        <TabPanel>
          <Canvas graph={data} context={context} stimulate={stimulate} />
        </TabPanel>
      </Tabs>
    </div>
  );
};
