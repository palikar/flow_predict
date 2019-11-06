#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Partitioned Unstructured Grid Reader'
c_1_solution0 = XMLPartitionedUnstructuredGridReader(FileName=['<file_place>'])

c_1_solution0.CellArrayStatus = ['Material Id', '_remote_index_', '_sub_domain_']
c_1_solution0.PointArrayStatus = ['p', 'u', 'v']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# set active source
SetActiveSource(c_1_solution0)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
renderView1.ViewSize = [1567, 782]

# get color transfer function/color map for 'a_sub_domain_'
a_sub_domain_LUT = GetColorTransferFunction('a_sub_domain_')
a_sub_domain_LUT.LockDataRange = 0
a_sub_domain_LUT.InterpretValuesAsCategories = 0
a_sub_domain_LUT.ShowCategoricalColorsinDataRangeOnly = 0
a_sub_domain_LUT.RescaleOnVisibilityChange = 0
a_sub_domain_LUT.EnableOpacityMapping = 0
a_sub_domain_LUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 1.5, 0.865003, 0.865003, 0.865003, 3.0, 0.705882, 0.0156863, 0.14902]
a_sub_domain_LUT.UseLogScale = 0
a_sub_domain_LUT.ColorSpace = 'Diverging'
a_sub_domain_LUT.UseBelowRangeColor = 0
a_sub_domain_LUT.BelowRangeColor = [0.0, 0.0, 0.0]
a_sub_domain_LUT.UseAboveRangeColor = 0
a_sub_domain_LUT.AboveRangeColor = [1.0, 1.0, 1.0]
a_sub_domain_LUT.NanColor = [1.0, 1.0, 0.0]
a_sub_domain_LUT.Discretize = 1
a_sub_domain_LUT.NumberOfTableValues = 256
a_sub_domain_LUT.ScalarRangeInitialized = 1.0
a_sub_domain_LUT.HSVWrap = 0
a_sub_domain_LUT.VectorComponent = 0
a_sub_domain_LUT.VectorMode = 'Magnitude'
a_sub_domain_LUT.AllowDuplicateScalars = 1
a_sub_domain_LUT.Annotations = []
a_sub_domain_LUT.ActiveAnnotatedValues = []
a_sub_domain_LUT.IndexedColors = []

# show data in view
c_1_solution0Display = Show(c_1_solution0, renderView1)
# trace defaults for the display properties.
c_1_solution0Display.Representation = 'Surface'
c_1_solution0Display.AmbientColor = [1.0, 1.0, 1.0]
c_1_solution0Display.ColorArrayName = ['CELLS', '_sub_domain_']
c_1_solution0Display.DiffuseColor = [1.0, 1.0, 1.0]
c_1_solution0Display.LookupTable = a_sub_domain_LUT
c_1_solution0Display.MapScalars = 1
c_1_solution0Display.InterpolateScalarsBeforeMapping = 1
c_1_solution0Display.Opacity = 1.0
c_1_solution0Display.PointSize = 2.0
c_1_solution0Display.LineWidth = 1.0
c_1_solution0Display.Interpolation = 'Gouraud'
c_1_solution0Display.Specular = 0.0
c_1_solution0Display.SpecularColor = [1.0, 1.0, 1.0]
c_1_solution0Display.SpecularPower = 100.0
c_1_solution0Display.Ambient = 0.0
c_1_solution0Display.Diffuse = 1.0
c_1_solution0Display.EdgeColor = [0.0, 0.0, 0.5]
c_1_solution0Display.BackfaceRepresentation = 'Follow Frontface'
c_1_solution0Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
c_1_solution0Display.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
c_1_solution0Display.BackfaceOpacity = 1.0
c_1_solution0Display.Position = [0.0, 0.0, 0.0]
c_1_solution0Display.Scale = [1.0, 1.0, 1.0]
c_1_solution0Display.Orientation = [0.0, 0.0, 0.0]
c_1_solution0Display.Origin = [0.0, 0.0, 0.0]
c_1_solution0Display.Pickable = 1
c_1_solution0Display.Texture = None
c_1_solution0Display.Triangulate = 0
c_1_solution0Display.NonlinearSubdivisionLevel = 1
c_1_solution0Display.UseDataPartitions = 0
c_1_solution0Display.OSPRayUseScaleArray = 0
c_1_solution0Display.OSPRayScaleArray = '_sub_domain_'
c_1_solution0Display.OSPRayScaleFunction = 'PiecewiseFunction'
c_1_solution0Display.Orient = 0
c_1_solution0Display.OrientationMode = 'Direction'
c_1_solution0Display.SelectOrientationVectors = 'None'
c_1_solution0Display.Scaling = 0
c_1_solution0Display.ScaleMode = 'No Data Scaling Off'
c_1_solution0Display.ScaleFactor = 0.22000000000000003
c_1_solution0Display.SelectScaleArray = '_sub_domain_'
c_1_solution0Display.GlyphType = 'Arrow'
c_1_solution0Display.SelectionCellLabelBold = 0
c_1_solution0Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
c_1_solution0Display.SelectionCellLabelFontFamily = 'Arial'
c_1_solution0Display.SelectionCellLabelFontSize = 18
c_1_solution0Display.SelectionCellLabelItalic = 0
c_1_solution0Display.SelectionCellLabelJustification = 'Left'
c_1_solution0Display.SelectionCellLabelOpacity = 1.0
c_1_solution0Display.SelectionCellLabelShadow = 0
c_1_solution0Display.SelectionPointLabelBold = 0
c_1_solution0Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
c_1_solution0Display.SelectionPointLabelFontFamily = 'Arial'
c_1_solution0Display.SelectionPointLabelFontSize = 18
c_1_solution0Display.SelectionPointLabelItalic = 0
c_1_solution0Display.SelectionPointLabelJustification = 'Left'
c_1_solution0Display.SelectionPointLabelOpacity = 1.0
c_1_solution0Display.SelectionPointLabelShadow = 0
c_1_solution0Display.ScalarOpacityUnitDistance = 0.10829656031906774
c_1_solution0Display.SelectMapper = 'Projected tetra'
c_1_solution0Display.GaussianRadius = 0.11000000000000001
c_1_solution0Display.ShaderPreset = 'Sphere'
c_1_solution0Display.Emissive = 0
c_1_solution0Display.ScaleByArray = 0
c_1_solution0Display.SetScaleArray = ['POINTS', 'p']
c_1_solution0Display.ScaleTransferFunction = 'PiecewiseFunction'
c_1_solution0Display.OpacityByArray = 0
c_1_solution0Display.OpacityArray = ['POINTS', 'p']
c_1_solution0Display.OpacityTransferFunction = 'PiecewiseFunction'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
c_1_solution0Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'Arrow' selected for 'GlyphType'
c_1_solution0Display.GlyphType.TipResolution = 6
c_1_solution0Display.GlyphType.TipRadius = 0.1
c_1_solution0Display.GlyphType.TipLength = 0.35
c_1_solution0Display.GlyphType.ShaftResolution = 6
c_1_solution0Display.GlyphType.ShaftRadius = 0.03
c_1_solution0Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
c_1_solution0Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
c_1_solution0Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# show color bar/color legend
c_1_solution0Display.SetScalarBarVisibility(renderView1, True)

# reset view to fit data
renderView1.ResetCamera()

# get opacity transfer function/opacity map for 'a_sub_domain_'
a_sub_domain_PWF = GetOpacityTransferFunction('a_sub_domain_')
a_sub_domain_PWF.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
a_sub_domain_PWF.AllowDuplicateScalars = 1
a_sub_domain_PWF.ScalarRangeInitialized = 1

# set scalar coloring
ColorBy(c_1_solution0Display, ('POINTS', '<var>'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(a_sub_domain_LUT, renderView1)

# rescale color and/or opacity maps used to include current data range
c_1_solution0Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
c_1_solution0Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'u'
uLUT = GetColorTransferFunction('<var>')
uLUT.LockDataRange = 0
uLUT.InterpretValuesAsCategories = 0
uLUT.ShowCategoricalColorsinDataRangeOnly = 0
uLUT.RescaleOnVisibilityChange = 0
uLUT.EnableOpacityMapping = 0
uLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 0.95, 0.865003, 0.865003, 0.865003, 1.9, 0.705882, 0.0156863, 0.14902]
uLUT.UseLogScale = 0
uLUT.ColorSpace = 'Diverging'
uLUT.UseBelowRangeColor = 0
uLUT.BelowRangeColor = [0.0, 0.0, 0.0]
uLUT.UseAboveRangeColor = 0
uLUT.AboveRangeColor = [1.0, 1.0, 1.0]
uLUT.NanColor = [1.0, 1.0, 0.0]
uLUT.Discretize = 1
uLUT.NumberOfTableValues = 256
uLUT.ScalarRangeInitialized = 1.0
uLUT.HSVWrap = 0
uLUT.VectorComponent = 0
uLUT.VectorMode = 'Magnitude'
uLUT.AllowDuplicateScalars = 1
uLUT.Annotations = []
uLUT.ActiveAnnotatedValues = []
uLUT.IndexedColors = []

# get opacity transfer function/opacity map for 'u'
uPWF = GetOpacityTransferFunction('<var>')
uPWF.Points = [0.0, 0.0, 0.5, 0.0, 1.9, 1.0, 0.5, 0.0]
uPWF.AllowDuplicateScalars = 1
uPWF.ScalarRangeInitialized = 1

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
uLUT.ApplyPreset('Grayscale', True)

# hide color bar/color legend
c_1_solution0Display.SetScalarBarVisibility(renderView1, False)

# reset view to fit data
renderView1.ResetCamera()

# reset view to fit data bounds
renderView1.ResetCamera(0.0, 2.2, 0.0, 0.41, 0.0, 0.0)

# current camera placement for renderView1
renderView1.CameraPosition = [1.1, 0.205, 2.016827658506227]
renderView1.CameraFocalPoint = [1.1, 0.205, 0.0]
renderView1.CameraParallelScale = 0.8189392298065164

# save animation images/movie
WriteAnimation('<output_place>', Magnification=1, FrameRate=15.0, Compression=True)

#### saving camera placements for all active views
renderView1.CameraPosition = [1.1, 0.205, 2.016827658506227]
renderView1.CameraFocalPoint = [1.1, 0.205, 0.0]
renderView1.CameraParallelScale = 1.1189392298065164

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
