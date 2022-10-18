import axios from 'axios';
import { useState } from 'react';
import { useEffect } from 'react';
import * as d3 from 'd3';
import { useRef } from 'react';
import { registerListener, unregisterListener } from '../events';

export default (props) => {
  const { context, threshold, stimulate } = props;

  const containerRef = useRef(null);
  const [data, setData] = useState(null);
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

  function processTopTokens(nodes) {
    nodes.sort((a, b) => b.s - a.s);

    setTopTokens(nodes.slice(0, 10));
  }

  function onClick() {
    const elems = d3.select(this).data();
    if (elems && elems.length > 0) {
      const { id } = elems[0];
      stimulate(context, id);
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
    line
      .attr('stroke', 'rgba(0,0,0,1)')
      .attr('stroke-width', (d) => d.weight)
      .attr('stroke-opacity', (d) => {
        return threshold > 0
          ? Math.min(d.source.s, d.target.s) - threshold >= 0
            ? 0.5
            : 0
          : 0.5;
      });
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
          threshold > 0
            ? Math.min(d.source.s, d.target.s) - threshold >= 0
              ? 1
              : 0
            : 1
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
            d3.zoomIdentity.translate(width / 2, height / 2).scale(1.25)
          );

        const simulation = d3
          .forceSimulation()
          .force('charge', d3.forceManyBody().strength(-30))
          .force(
            'link',
            d3
              .forceLink()
              .id((d) => d.id)
              .distance(30)
          )
          // .force('x', d3.forceX())
          // .force('y', d3.forceY())
          .force('center', d3.forceCenter(width / 2, height / 2));

        var link = svg
          .append('g')
          .attr('class', 'links')
          .selectAll('line')
          .data(graph.links)
          .enter()
          .append('line');

        var node = svg
          .append('g')
          .attr('class', 'nodes')
          .selectAll('g')
          .data(graph.nodes)
          .enter()
          .append('g');

        setNodes(node);
        setLinks(link);

        setTimeout(() => {
          doNode(node);
          doLine(link);
        }, 10);

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
        processTopTokens(graph.nodes);
        if (nodes) {
          const old = new Map(nodes.data().map((d) => [d.id, d]));
          const updatedNodes = graph.nodes.map((d) => {
            const result = Object.assign(old.get(d.id) || {}, d);
            result.s = d.s;
            return result;
          });
          const updatedLinks = graph.links.map((d) => Object.assign({}, d));
          sim.nodes(updatedNodes);
          sim.force('link').links(updatedLinks);
          sim.alpha(0.001).restart();

          const newNode = nodes
            .data(updatedNodes, (d) => d.id)
            .join((enter) => {
              enter = enter.append('g');

              enter.on('click', onClick);

              enter.attr('class', 'node');
              doNode(enter);

              return enter;
            });

          // const newLinks = links
          //   .data(graph.links, (d) => {
          //     d.source = updatedNodes.find((n) => n.id === d.source);
          //     d.target = updatedNodes.find((n) => n.id === d.target);
          //   })
          //   .join('line');

          // doLine(newLinks);
          setNodes(newNode);
          // setLinks(newLinks);
          sim.alpha(0.001).restart();
          adjustOpacity();
        }
      });
  }, [state]);

  return (
    <svg className="svg" width="100%" height="100%" ref={containerRef}></svg>
  );
};
