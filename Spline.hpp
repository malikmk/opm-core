// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*****************************************************************************
 *   Copyright (C) 2010-2013 by Andreas Lauser                               *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/*!
 * \file
 * \copydoc Opm::Spline
 */
#ifndef OPM_SPLINE_HH
#define OPM_SPLINE_HH

#include <opm/core/utility/TridiagonalMatrix.hpp>
#include <opm/core/utility/PolynomialUtils.hpp>
#include <opm/core/utility/ErrorMacros.hpp>

#include <ostream>
#include <vector>
#include <tuple>

namespace Opm
{
/*!
 * \brief Class implementing cubic splines.
 *
 * This class supports full, natural, periodic and monotonic cubic
 * splines.
 *
 * Full a splines \f$s(x)\f$ are splines which, given \f$n\f$ sampling
 * points \f$x_1, \dots, x_n\f$, fulfill the following conditions
 * \f{align*}{
 *  s(x_i)   & = y_i \quad \forall i \in \{1, \dots, n \} \\
 *  s'(x_1)  & = m_1 \\
 *  s'(x_n)  & = m_n
 * \f}
 * for any given boundary slopes \f$m_1\f$ and \f$m_n\f$.
 *
 * Natural splines which are defined by
 *\f{align*}{
 *    s(x_i)     & = y_i \quad \forall i \in \{1, \dots, n \} \\
 *    s''(x_1)   & = 0 \\
 *    s''(x_n)   & = 0
 *\f}
 *
 * For periodic splines of splines the slopes at the boundaries are identical:
 *\f{align*}{
 *    s(x_i)     & = y_i \quad \forall i \in \{1, \dots, n \} \\
 *    s'(x_1)    & = s'(x_n) \\
 *    s''(x_1)   & = s''(x_n) \;.
 *\f}
 *
 * Finally, there are monotonic splines which guarantee that the curve
 * is confined by its sampling points, i.e.,
 * \f[
 y_i \leq s(x) \leq y_{i+1} \quad \text{for} x_i \leq x \leq x_{i+1} \;.
 * \f]
 * For more information on monotonic splines, see
 * http://en.wikipedia.org/wiki/Monotone_cubic_interpolation
 *
 * Full, natural and periodic splines are continuous in their first
 * and second derivatives, i.e.,
 * \f[
 s \in \mathcal{C}^2
 * \f]
 * holds for such splines. Monotonic splines are only continuous up to
 * their first derivative, i.e., for these only
 * \f[
 s \in \mathcal{C}^1
 * \f]
 * is true.
 */
template<class Scalar>
class Spline
{
    typedef Opm::TridiagonalMatrix<Scalar> Matrix;
    typedef std::vector<Scalar> Vector;

public:
    /*!
     * \brief The type of the spline to be created.
     *
     * \copydetails Spline
     */
    enum SplineType {
        Full,
        Natural,
        Periodic,
        Monotonic
    };

    /*!
     * \brief Default constructor for a spline.
     *
     * To specfiy the acutal curve, use one of the set() methods.
     */
    Spline()
    { }

    /*!
     * \brief Convenience constructor for a full spline with just two sampling points
     *
     * \param x0 The \f$x\f$ value of the first sampling point
     * \param x1 The \f$x\f$ value of the second sampling point
     * \param y0 The \f$y\f$ value of the first sampling point
     * \param y1 The \f$y\f$ value of the second sampling point
     * \param m0 The slope of the spline at \f$x_0\f$
     * \param m1 The slope of the spline at \f$x_1\f$
     */
    Spline(Scalar x0, Scalar x1,
           Scalar y0, Scalar y1,
           Scalar m0, Scalar m1)
    { set(x0, x1, y0, y1, m0, m1); }

    /*!
     * \brief Convenience constructor for a natural or a periodic spline
     *
     * \param nSamples The number of sampling points (must be > 2)
     * \param x An array containing the \f$x\f$ values of the spline's sampling points
     * \param y An array containing the \f$y\f$ values of the spline's sampling points
     * \param periodic Indicates whether a natural or a periodic spline should be created
     */
    template <class ScalarArrayX, class ScalarArrayY>
    Spline(int nSamples,
           const ScalarArrayX &x,
           const ScalarArrayY &y,
           SplineType splineType = Natural,
           bool sortInputs = false)
    { this->setXYArrays(nSamples, x, y, splineType, sortInputs); }

    /*!
     * \brief Convenience constructor for a natural or a periodic spline
     *
     * \param nSamples The number of sampling points (must be > 2)
     * \param points An array of \f$(x,y)\f$ tuples of the spline's sampling points
     * \param periodic Indicates whether a natural or a periodic spline should be created
     */
    template <class PointArray>
    Spline(int nSamples,
           const PointArray &points,
           SplineType splineType = Natural,
           bool sortInputs = false)
    { this->setArrayOfPoints(nSamples, points, splineType, sortInputs); }

    /*!
     * \brief Convenience constructor for a natural or a periodic spline
     *
     * \param x An array containing the \f$x\f$ values of the spline's sampling points (must have a size() method)
     * \param y An array containing the \f$y\f$ values of the spline's sampling points (must have a size() method)
     * \param periodic Indicates whether a natural or a periodic spline should be created
     */
    template <class ScalarContainer>
    Spline(const ScalarContainer &x,
           const ScalarContainer &y,
           SplineType splineType = Natural,
           bool sortInputs = false)
    { this->setXYContainers(x, y, splineType, sortInputs); }

    /*!
     * \brief Convenience constructor for a natural or a periodic spline
     *
     * \param points An array of \f$(x,y)\f$ tuples of the spline's sampling points (must have a size() method)
     * \param periodic Indicates whether a natural or a periodic spline should be created
     */
    template <class PointContainer>
    Spline(const PointContainer &points,
           SplineType splineType = Natural,
           bool sortInputs = false)
    { this->setContainerOfPoints(points, splineType, sortInputs); }

    /*!
     * \brief Convenience constructor for a full spline
     *
     * \param nSamples The number of sampling points (must be >= 2)
     * \param x An array containing the \f$x\f$ values of the spline's sampling points
     * \param y An array containing the \f$y\f$ values of the spline's sampling points
     * \param m0 The slope of the spline at \f$x_0\f$
     * \param m1 The slope of the spline at \f$x_n\f$
     * \param sortInputs Indicates whether the sample points should be sorted (this is not necessary if they are already sorted in ascending or descending order)
     */
    template <class ScalarArray>
    Spline(int nSamples,
           const ScalarArray &x,
           const ScalarArray &y,
           Scalar m0,
           Scalar m1,
           bool sortInputs = false)
    { this->setXYArrays(nSamples, x, y, m0, m1, sortInputs); }

    /*!
     * \brief Convenience constructor for a full spline
     *
     * \param nSamples The number of sampling points (must be >= 2)
     * \param points An array containing the \f$x\f$ and \f$x\f$ values of the spline's sampling points
     * \param m0 The slope of the spline at \f$x_0\f$
     * \param m1 The slope of the spline at \f$x_n\f$
     * \param sortInputs Indicates whether the sample points should be sorted (this is not necessary if they are already sorted in ascending or descending order)
     */
    template <class PointArray>
    Spline(int nSamples,
           const PointArray &points,
           Scalar m0,
           Scalar m1,
           bool sortInputs = false)
    { this->setArrayOfPoints(nSamples, points, m0, m1, sortInputs); }

    /*!
     * \brief Convenience constructor for a full spline
     *
     * \param x An array containing the \f$x\f$ values of the spline's sampling points (must have a size() method)
     * \param y An array containing the \f$y\f$ values of the spline's sampling points (must have a size() method)
     * \param m0 The slope of the spline at \f$x_0\f$
     * \param m1 The slope of the spline at \f$x_n\f$
     * \param sortInputs Indicates whether the sample points should be sorted (this is not necessary if they are already sorted in ascending or descending order)
     */
    template <class ScalarContainerX, class ScalarContainerY>
    Spline(const ScalarContainerX &x,
           const ScalarContainerY &y,
           Scalar m0,
           Scalar m1,
           bool sortInputs = false)
    { this->setXYContainers(x, y, m0, m1, sortInputs); }

    /*!
     * \brief Convenience constructor for a full spline
     *
     * \param points An array of \f$(x,y)\f$ tuples of the spline's sampling points (must have a size() method)
     * \param m0 The slope of the spline at \f$x_0\f$
     * \param m1 The slope of the spline at \f$x_n\f$
     * \param sortInputs Indicates whether the sample points should be sorted (this is not necessary if they are already sorted in ascending or descending order)
     */
    template <class PointContainer>
    Spline(const PointContainer &points,
           Scalar m0,
           Scalar m1,
           bool sortInputs = false)
    { this->setContainerOfPoints(points, m0, m1, sortInputs); }

    /*!
     * \brief Returns the number of sampling points.
     */
    int numSamples() const
    { return xPos_.size(); }

    /*!
     * \brief Set the sampling points and the boundary slopes of the
     *        spline with two sampling points.
     *
     * \param x0 The \f$x\f$ value of the first sampling point
     * \param x1 The \f$x\f$ value of the second sampling point
     * \param y0 The \f$y\f$ value of the first sampling point
     * \param y1 The \f$y\f$ value of the second sampling point
     * \param m0 The slope of the spline at \f$x_0\f$
     * \param m1 The slope of the spline at \f$x_1\f$
     */
    void set(Scalar x0, Scalar x1,
             Scalar y0, Scalar y1,
             Scalar m0, Scalar m1)
    {
        slopeVec_.resize(2);
        xPos_.resize(2);
        yPos_.resize(2);

        if (x0 > x1) {
            xPos_[0] = x1;
            xPos_[1] = x0;
            yPos_[0] = y1;
            yPos_[1] = y0;
        }
        else {
            xPos_[0] = x0;
            xPos_[1] = x1;
            yPos_[0] = y0;
            yPos_[1] = y1;
        }

        slopeVec_[0] = m0;
        slopeVec_[1] = m1;

        Matrix M(numSamples());
        Vector d(numSamples());
        Vector moments(numSamples());
        this->makeFullSystem_(M, d, m0, m1);

        // solve for the moments
        M.solve(moments, d);

        this->setSlopesFromMoments_(slopeVec_, moments);
    }


    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    // Full splines                      //
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////

    /*!
     * \brief Set the sampling points and the boundary slopes of a
     *        full spline using C-style arrays.
     *
     * This method uses separate array-like objects for the values of
     * the X and Y coordinates. In this context 'array-like' means
     * that an access to the members is provided via the []
     * operator. (e.g. C arrays, std::vector, std::array, etc.)  Each
     * array must be of size 'nSamples' at least. Also, the number of
     * sampling points must be larger than 1.
     */
    template <class ScalarArrayX, class ScalarArrayY>
    void setXYArrays(int nSamples,
                     const ScalarArrayX &x,
                     const ScalarArrayY &y,
                     Scalar m0, Scalar m1,
                     bool sortInputs = false)
    {
        assert(nSamples > 1);

        setNumSamples_(nSamples);
        for (int i = 0; i < nSamples; ++i) {
            xPos_[i] = x[i];
            yPos_[i] = y[i];
        }

        if (sortInputs)
            sortInput_();
        else if (xPos_[0] > xPos_[numSamples() - 1])
            reverseSamplingPoints_();

        makeFullSpline_(m0, m1);
    }

    /*!
     * \brief Set the sampling points and the boundary slopes of a
     *        full spline using STL-compatible containers.
     *
     * This method uses separate STL-compatible containers for the
     * values of the sampling points' X and Y
     * coordinates. "STL-compatible" means that the container provides
     * access to iterators using the begin(), end() methods and also
     * provides a size() method. Also, the number of entries in the X
     * and the Y containers must be equal and larger than 1.
     */
    template <class ScalarContainerX, class ScalarContainerY>
    void setXYContainers(const ScalarContainerX &x,
                         const ScalarContainerY &y,
                         Scalar m0, Scalar m1,
                         bool sortInputs = false)
    {
        assert(x.size() == y.size());
        assert(x.size() > 1);

        setNumSamples_(x.size());

        std::copy(x.begin(), x.end(), xPos_.begin());
        std::copy(y.begin(), y.end(), yPos_.begin());

        if (sortInputs)
            sortInput_();
        else if (xPos_[0] > xPos_[numSamples() - 1])
            reverseSamplingPoints_();

        makeFullSpline_(m0, m1);
    }

    /*!
     * \brief Set the sampling points and the boundary slopes of a
     *        full spline using a C-style array.
     *
     * This method uses a single array of sampling points, which are
     * seen as an array-like object which provides access to the X and
     * Y coordinates.  In this context 'array-like' means that an
     * access to the members is provided via the [] operator. (e.g. C
     * arrays, std::vector, std::array, etc.)  The array containing
     * the sampling points must be of size 'nSamples' at least. Also,
     * the number of sampling points must be larger than 1.
     */
    template <class PointArray>
    void setArrayOfPoints(int nSamples,
                          const PointArray &points,
                          Scalar m0,
                          Scalar m1,
                          bool sortInputs = false)
    {
        // a spline with no or just one sampling points? what an
        // incredible bad idea!
        assert(nSamples > 1);

        setNumSamples_(nSamples);
        for (int i = 0; i < nSamples; ++i) {
            xPos_[i] = points[i][0];
            yPos_[i] = points[i][1];
        }

        if (sortInputs)
            sortInput_();
        else if (xPos_[0] > xPos_[numSamples() - 1])
            reverseSamplingPoints_();

        makeFullSpline_(m0, m1);
    }

    /*!
     * \brief Set the sampling points and the boundary slopes of a
     *        full spline using a STL-compatible container of
     *        array-like objects.
     *
     * This method uses a single STL-compatible container of sampling
     * points, which are assumed to be array-like objects storing the
     * X and Y coordinates.  "STL-compatible" means that the container
     * provides access to iterators using the begin(), end() methods
     * and also provides a size() method. Also, the number of entries
     * in the X and the Y containers must be equal and larger than 1.
     */
    template <class XYContainer>
    void setContainerOfPoints(const XYContainer &points,
                              Scalar m0,
                              Scalar m1,
                              bool sortInputs = false)
    {
        // a spline with no or just one sampling points? what an
        // incredible bad idea!
        assert(points.size() > 1);

        setNumSamples_(points.size());
        typename XYContainer::const_iterator it = points.begin();
        typename XYContainer::const_iterator endIt = points.end();
        for (int i = 0; it != endIt; ++i, ++it) {
            xPos_[i] = (*it)[0];
            yPos_[i] = (*it)[1];
        }

        if (sortInputs)
            sortInput_();
        else if (xPos_[0] > xPos_[numSamples() - 1])
            reverseSamplingPoints_();

        // make a full spline
        makeFullSpline_(m0, m1);
    }

    /*!
     * \brief Set the sampling points and the boundary slopes of a
     *        full spline using a STL-compatible container of
     *        tuple-like objects.
     *
     * This method uses a single STL-compatible container of sampling
     * points, which are assumed to be tuple-like objects storing the
     * X and Y coordinates.  "tuple-like" means that the objects
     * provide access to the x values via std::get<0>(obj) and to the
     * y value via std::get<1>(obj) (e.g. std::tuple or
     * std::pair). "STL-compatible" means that the container provides
     * access to iterators using the begin(), end() methods and also
     * provides a size() method. Also, the number of entries in the X
     * and the Y containers must be equal and larger than 1.
     */
    template <class XYContainer>
    void setContainerOfTuples(const XYContainer &points,
                              Scalar m0,
                              Scalar m1,
                              bool sortInputs = false)
    {
        // resize internal arrays
        setNumSamples_(points.size());
        typename XYContainer::const_iterator it = points.begin();
        typename XYContainer::const_iterator endIt = points.end();
        for (int i = 0; it != endIt; ++i, ++it) {
            xPos_[i] = std::get<0>(*it);
            yPos_[i] = std::get<1>(*it);
        }

        if (sortInputs)
            sortInput_();
        else if (xPos_[0] > xPos_[numSamples() - 1])
            reverseSamplingPoints_();

        // make a full spline
        makeFullSpline_(m0, m1);
    }

    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    // Natural/Periodic splines          //
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    /*!
     * \brief Set the sampling points natural spline using C-style arrays.
     *
     * This method uses separate array-like objects for the values of
     * the X and Y coordinates. In this context 'array-like' means
     * that an access to the members is provided via the []
     * operator. (e.g. C arrays, std::vector, std::array, etc.)  Each
     * array must be of size 'nSamples' at least. Also, the number of
     * sampling points must be larger than 1.
     */
    template <class ScalarArrayX, class ScalarArrayY>
    void setXYArrays(int nSamples,
                     const ScalarArrayX &x,
                     const ScalarArrayY &y,
                     SplineType splineType = Natural,
                     bool sortInputs = false)
    {
        assert(nSamples > 1);

        setNumSamples_(nSamples);
        for (int i = 0; i < nSamples; ++i) {
            xPos_[i] = x[i];
            yPos_[i] = y[i];
        }

        if (sortInputs)
            sortInput_();
        else if (xPos_[0] > xPos_[numSamples() - 1])
            reverseSamplingPoints_();

        if (splineType == Periodic)
            makePeriodicSpline_();
        else if (splineType == Natural)
            makeNaturalSpline_();
        else if (splineType == Monotonic)
            this->makeMonotonicSpline_(slopeVec_);
        else
            OPM_THROW(std::runtime_error, "Spline type " << splineType << " not supported at this place");
    }

    /*!
     * \brief Set the sampling points of a natural spline using
     *        STL-compatible containers.
     *
     * This method uses separate STL-compatible containers for the
     * values of the sampling points' X and Y
     * coordinates. "STL-compatible" means that the container provides
     * access to iterators using the begin(), end() methods and also
     * provides a size() method. Also, the number of entries in the X
     * and the Y containers must be equal and larger than 1.
     */
    template <class ScalarContainerX, class ScalarContainerY>
    void setXYContainers(const ScalarContainerX &x,
                         const ScalarContainerY &y,
                         SplineType splineType = Natural,
                         bool sortInputs = false)
    {
        assert(x.size() == y.size());
        assert(x.size() > 1);

        setNumSamples_(x.size());
        std::copy(x.begin(), x.end(), xPos_.begin());
        std::copy(y.begin(), y.end(), yPos_.begin());

        if (sortInputs)
            sortInput_();
        else if (xPos_[0] > xPos_[numSamples() - 1])
            reverseSamplingPoints_();

        if (splineType == Periodic)
            makePeriodicSpline_();
        else if (splineType == Natural)
            makeNaturalSpline_();
        else if (splineType == Monotonic)
            this->makeMonotonicSpline_(slopeVec_);
        else
            OPM_THROW(std::runtime_error, "Spline type " << splineType << " not supported at this place");
    }

    /*!
     * \brief Set the sampling points of a natural spline using a
     *        C-style array.
     *
     * This method uses a single array of sampling points, which are
     * seen as an array-like object which provides access to the X and
     * Y coordinates.  In this context 'array-like' means that an
     * access to the members is provided via the [] operator. (e.g. C
     * arrays, std::vector, std::array, etc.)  The array containing
     * the sampling points must be of size 'nSamples' at least. Also,
     * the number of sampling points must be larger than 1.
     */
    template <class PointArray>
    void setArrayOfPoints(int nSamples,
                          const PointArray &points,
                          SplineType splineType = Natural,
                          bool sortInputs = false)
    {
        // a spline with no or just one sampling points? what an
        // incredible bad idea!
        assert(nSamples > 1);

        setNumSamples_(nSamples);
        for (int i = 0; i < nSamples; ++i) {
            xPos_[i] = points[i][0];
            yPos_[i] = points[i][1];
        }

        if (sortInputs)
            sortInput_();
        else if (xPos_[0] > xPos_[numSamples() - 1])
            reverseSamplingPoints_();

        if (splineType == Periodic)
            makePeriodicSpline_();
        else if (splineType == Natural)
            makeNaturalSpline_();
        else if (splineType == Monotonic)
            this->makeMonotonicSpline_(slopeVec_);
        else
            OPM_THROW(std::runtime_error, "Spline type " << splineType << " not supported at this place");
    }

    /*!
     * \brief Set the sampling points of a natural spline using a
     *        STL-compatible container of array-like objects.
     *
     * This method uses a single STL-compatible container of sampling
     * points, which are assumed to be array-like objects storing the
     * X and Y coordinates.  "STL-compatible" means that the container
     * provides access to iterators using the begin(), end() methods
     * and also provides a size() method. Also, the number of entries
     * in the X and the Y containers must be equal and larger than 1.
     */
    template <class XYContainer>
    void setContainerOfPoints(const XYContainer &points,
                              SplineType splineType = Natural,
                              bool sortInputs = false)
    {
        // a spline with no or just one sampling points? what an
        // incredible bad idea!
        assert(points.size() > 1);

        setNumSamples_(points.size());
        typename XYContainer::const_iterator it = points.begin();
        typename XYContainer::const_iterator endIt = points.end();
        for (int i = 0; it != endIt; ++ i, ++it) {
            xPos_[i] = (*it)[0];
            yPos_[i] = (*it)[1];
        }

        if (sortInputs)
            sortInput_();
        else if (xPos_[0] > xPos_[numSamples() - 1])
            reverseSamplingPoints_();

        if (splineType == Periodic)
            makePeriodicSpline_();
        else if (splineType == Natural)
            makeNaturalSpline_();
        else if (splineType == Monotonic)
            this->makeMonotonicSpline_(slopeVec_);
        else
            OPM_THROW(std::runtime_error, "Spline type " << splineType << " not supported at this place");
    }

    /*!
     * \brief Set the sampling points of a natural spline using a
     *        STL-compatible container of tuple-like objects.
     *
     * This method uses a single STL-compatible container of sampling
     * points, which are assumed to be tuple-like objects storing the
     * X and Y coordinates.  "tuple-like" means that the objects
     * provide access to the x values via std::get<0>(obj) and to the
     * y value via std::get<1>(obj) (e.g. std::tuple or
     * std::pair). "STL-compatible" means that the container provides
     * access to iterators using the begin(), end() methods and also
     * provides a size() method. Also, the number of entries in the X
     * and the Y containers must be equal and larger than 1.
     */
    template <class XYContainer>
    void setContainerOfTuples(const XYContainer &points,
                              SplineType splineType = Natural,
                              bool sortInputs = false)
    {
        // resize internal arrays
        setNumSamples_(points.size());
        typename XYContainer::const_iterator it = points.begin();
        typename XYContainer::const_iterator endIt = points.end();
        for (int i = 0; it != endIt; ++i, ++it) {
            xPos_[i] = std::get<0>(*it);
            yPos_[i] = std::get<1>(*it);
        }

        if (sortInputs)
            sortInput_();
        else if (xPos_[0] > xPos_[numSamples() - 1])
            reverseSamplingPoints_();

        if (splineType == Periodic)
            makePeriodicSpline_();
        else if (splineType == Natural)
            makeNaturalSpline_();
        else if (splineType == Monotonic)
            this->makeMonotonicSpline_(slopeVec_);
        else
            OPM_THROW(std::runtime_error, "Spline type " << splineType << " not supported at this place");
    }

    /*!
     * \brief Return true iff the given x is in range [x1, xn].
     */
    bool applies(Scalar x) const
    {
        return x_(0) <= x && x <= x_(numSamples() - 1);
    }

    /*!
     * \brief Return the x value of the leftmost sampling point.
     */
    Scalar xMin() const
    { return x_(0); }

    /*!
     * \brief Return the x value of the rightmost sampling point.
     */
    Scalar xMax() const
    { return x_(numSamples() - 1); }

    /*!
     * \brief Prints k tuples of the format (x, y, dx/dy, isMonotonic)
     *        to stdout.
     *
     * If the spline does not apply for parts of [x0, x1] it is
     * extrapolated using a straight line. The result can be inspected
     * using the following commands:
     *
     ----------- snip -----------
     ./yourProgramm > spline.csv
     gnuplot

     gnuplot> plot "spline.csv" using 1:2 w l ti "Curve", \
     "spline.csv" using 1:3 w l ti "Derivative", \
     "spline.csv" using 1:4 w p ti "Monotonic"
     ----------- snap -----------
    */
    void printCSV(Scalar xi0, Scalar xi1, int k, std::ostream &os = std::cout) const
    {
        Scalar x0 = std::min(xi0, xi1);
        Scalar x1 = std::max(xi0, xi1);
        const int n = numSamples() - 1;
        for (int i = 0; i <= k; ++i) {
            double x = i*(x1 - x0)/k + x0;
            double x_p1 = x + (x1 - x0)/k;
            double y;
            double dy_dx;
            double mono = 1;
            if (!applies(x)) {
                if (x < x_(0)) {
                    dy_dx = evalDerivative(x_(0));
                    y = (x - x_(0))*dy_dx + y_(0);
                    mono = (dy_dx>0)?1:-1;
                }
                else if (x > x_(n)) {
                    dy_dx = evalDerivative(x_(n));
                    y = (x - x_(n))*dy_dx + y_(n);
                    mono = (dy_dx>0)?1:-1;
                }
                else {
                    OPM_THROW(std::runtime_error,
                              "The sampling points given to a spline must be sorted by their x value!");
                }
            }
            else {
                y = eval(x);
                dy_dx = evalDerivative(x);
                mono = monotonic(x, x_p1, /*extrapolate=*/true);
            }

            os << x << " " << y << " " << dy_dx << " " << mono << "\n";
        }
    }

    /*!
     * \brief Evaluate the spline at a given position.
     *
     * \param x The value on the abscissa where the spline ought to be
     *          evaluated
     * \param extrapolate If this parameter is set to true, the spline
     *                    will be extended beyond its range by
     *                    straight lines, if false calling extrapolate
     *                    for \f$ x \not [x_{min}, x_{max}]\f$ will
     *                    cause a failed assertation.
     */
    Scalar eval(Scalar x, bool extrapolate=false) const
    {
        assert(extrapolate || applies(x));

        // handle extrapolation
        if (extrapolate) {
            if (x < xMin()) {
                Scalar m = evalDerivative_(xMin(), /*segmentIdx=*/0);
                Scalar y0 = y_(0);
                return y0 + m*(x - xMin());
            }
            else if (x > xMax()) {
                Scalar m = evalDerivative_(xMax(), /*segmentIdx=*/numSamples()-2);
                Scalar y0 = y_(numSamples() - 1);
                return y0 + m*(x - xMax());
            }
        }

        return eval_(x, segmentIdx_(x));
    }

    /*!
     * \brief Evaluate the spline's derivative at a given position.
     *
     * \param x The value on the abscissa where the spline's
     *          derivative ought to be evaluated
     *
     * \param extrapolate If this parameter is set to true, the spline
     *                    will be extended beyond its range by
     *                    straight lines, if false calling extrapolate
     *                    for \f$ x \not [x_{min}, x_{max}]\f$ will
     *                    cause a failed assertation.
     */
    Scalar evalDerivative(Scalar x, bool extrapolate=false) const
    {
        assert(extrapolate || applies(x));
        if (extrapolate) {
            if (x < xMin())
                return evalDerivative_(xMin(), /*segmentIdx=*/0);
            else if (x > xMax())
                return evalDerivative_(xMax(), /*segmentIdx=*/numSamples() - 2);
        }

        return evalDerivative_(x, segmentIdx_(x));
    }

    /*!
     * \brief Evaluate the spline's second derivative at a given position.
     *
     * \param x The value on the abscissa where the spline's
     *          derivative ought to be evaluated
     *
     * \param extrapolate If this parameter is set to true, the spline
     *                    will be extended beyond its range by
     *                    straight lines, if false calling extrapolate
     *                    for \f$ x \not [x_{min}, x_{max}]\f$ will
     *                    cause a failed assertation.
     */
    Scalar evalSecondDerivative(Scalar x, bool extrapolate=false) const
    {
        assert(extrapolate || applies(x));
        if (extrapolate)
            return 0.0;

        return evalDerivative2_(x, segmentIdx_(x));
    }

    /*!
     * \brief Evaluate the spline's third derivative at a given position.
     *
     * \param x The value on the abscissa where the spline's
     *          derivative ought to be evaluated
     *
     * \param extrapolate If this parameter is set to true, the spline
     *                    will be extended beyond its range by
     *                    straight lines, if false calling extrapolate
     *                    for \f$ x \not [x_{min}, x_{max}]\f$ will
     *                    cause a failed assertation.
     */
    Scalar evalThirdDerivative(Scalar x, bool extrapolate=false) const
    {
        assert(extrapolate || applies(x));
        if (extrapolate)
            return 0.0;

        return evalDerivative3_(x, segmentIdx_(x));
    }

    /*!
     * \brief Find the intersections of the spline with a cubic
     *        polynomial in the whole interval, throws
     *        Opm::MathError exception if there is more or less than
     *        one solution.
     */
    Scalar intersect(Scalar a, Scalar b, Scalar c, Scalar d) const
    {
        return intersectInterval(xMin(), xMax(), a, b, c, d);
    }

    /*!
     * \brief Find the intersections of the spline with a cubic
     *        polynomial in a sub-interval of the spline, throws
     *        Opm::MathError exception if there is more or less than
     *        one solution.
     */
    Scalar intersectInterval(Scalar x0, Scalar x1,
                             Scalar a, Scalar b, Scalar c, Scalar d) const
    {
        assert(applies(x0) && applies(x1));

        Scalar tmpSol[3], sol = 0;
        int nSol = 0;
        int iFirst = segmentIdx_(x0);
        int iLast = segmentIdx_(x1);
        for (int i = iFirst; i <= iLast; ++i)
        {
            int nCur = intersectSegment_(tmpSol, i, a, b, c, d, x0, x1);
            if (nCur == 1)
                sol = tmpSol[0];

            nSol += nCur;
            if (nSol > 1) {
                OPM_THROW(std::runtime_error,
                          "Spline has more than one intersection"); //<<a<<"x^3 + "<<b<"x^2 + "<<c<"x + "<<d);
            }
        }

        if (nSol != 1)
            OPM_THROW(std::runtime_error,
                      "Spline has no intersection"); //<<a<"x^3 + " <<b<"x^2 + "<<c<"x + "<<d<<"!");

        return sol;
    }

    /*!
     * \brief Returns 1 if the spline is monotonically increasing, -1
     *        if the spline is mononously decreasing and 0 if the
     *        spline is not monotonous in the interval (x0, x1).
     *
     * In the corner case that the spline is constant within the given
     * interval, this method returns 3.
     */
    int monotonic(Scalar x0, Scalar x1, bool extrapolate=false) const
    {
        assert(x0 != x1);

        // make sure that x0 is smaller than x1
        if (x0 > x1)
            std::swap(x0, x1);

        assert(x0 < x1);

        int r = 3;
        if (x0 < xMin()) {
            static_cast<void>(extrapolate);
            assert(extrapolate);
            Scalar m = evalDerivative_(xMin(), /*segmentIdx=*/0);
            if (std::abs(m) < 1e-20)
                r = (m < 0)?-1:1;
            x0 = xMin();
        };

        int i = segmentIdx_(x0);
        if (x_(i + 1) >= x1) {
            // interval is fully contained within a single spline
            // segment
            monotonic_(i, x0, x1, r);
            return r;
        }

        // the first segment overlaps with the specified interval
        // partially
        monotonic_(i, x0, x_(i+1), r);
        ++ i;

        // make sure that the segments which are completly in the
        // interval [x0, x1] all exhibit the same monotonicity.
        int iEnd = segmentIdx_(x1);
        for (; i < iEnd - 1; ++i) {
            monotonic_(i, x_(i), x_(i + 1), r);
            if (!r)
                return 0;
        }

        // if the user asked for a part of the spline which is
        // extrapolated, we need to check the slope at the spline's
        // endpoint
        if (x1 > xMax()) {
            assert(extrapolate);

            Scalar m = evalDerivative_(xMax(), /*segmentIdx=*/numSamples() - 2);
            if (m < 0)
                return (r < 0 || r==3)?-1:0;
            else if (m > 0)
                return (r > 0 || r==3)?1:0;

            return r;
        }

        // check for the last segment
        monotonic_(iEnd, x_(iEnd), x1, r);

        return r;
    }

    /*!
     * \brief Same as monotonic(x0, x1), but with the entire range of the
     *        spline as interval.
     */
    int monotonic() const
    { return monotonic(xMin(), xMax()); }

protected:
    /*!
     * \brief Helper class needed to sort the input sampling points.
     */
    struct ComparatorX_
    {
        ComparatorX_(const std::vector<Scalar> &x)
            : x_(x)
        {};

        bool operator ()(int idxA, int idxB) const
        { return x_.at(idxA) < x_.at(idxB); }

        const std::vector<Scalar> &x_;
    };

    /*!
     * \brief Sort the sample points in ascending order of their x value.
     */
    void sortInput_()
    {
        int n = numSamples();

        // create a vector containing 0...n-1
        std::vector<int> idxVector(n);
        for (int i = 0; i < n; ++i)
            idxVector[i] = i;

        // sort the indices according to the x values of the sample
        // points
        ComparatorX_ cmp(xPos_);
        std::sort(idxVector.begin(), idxVector.end(), cmp);

        // reorder the sample points
        std::vector<Scalar> tmpX(n), tmpY(n);
        for (size_t i = 0; i < idxVector.size(); ++ i) {
            tmpX[i] = xPos_[idxVector[i]];
            tmpY[i] = yPos_[idxVector[i]];
        }
        xPos_ = tmpX;
        yPos_ = tmpY;
    }

    /*!
     * \brief Reverse order of the elements in the arrays which
     *        contain the sampling points.
     */
    void reverseSamplingPoints_()
    {
        // reverse the arrays
        int n = numSamples();
        for (int i = 0; i <= (n - 1)/2; ++i) {
            std::swap(xPos_[i], xPos_[n - i - 1]);
            std::swap(yPos_[i], yPos_[n - i - 1]);
        }
    }

    /*!
     * \brief Resizes the internal vectors to store the sample points.
     */
    void setNumSamples_(int nSamples)
    {
        xPos_.resize(nSamples);
        yPos_.resize(nSamples);
        slopeVec_.resize(nSamples);
    }

    /*!
     * \brief Create a natural spline from the already set sampling points.
     *
     * This creates a temporary matrix and right hand side vector.
     */
    void makeFullSpline_(Scalar m0, Scalar m1)
    {
        Matrix M(numSamples());
        std::vector<Scalar> d(numSamples());
        std::vector<Scalar> moments(numSamples());

        // create linear system of equations
        this->makeFullSystem_(M, d, m0, m1);

        // solve for the moments (-> second derivatives)
        M.solve(moments, d);

        // convert the moments to slopes at the sample points
        this->setSlopesFromMoments_(slopeVec_, moments);
    }

    /*!
     * \brief Create a natural spline from the already set sampling points.
     *
     * This creates a temporary matrix and right hand side vector.
     */
    void makeNaturalSpline_()
    {
        Matrix M(numSamples(), numSamples());
        Vector d(numSamples());
        Vector moments(numSamples());

        // create linear system of equations
        this->makeNaturalSystem_(M, d);

        // solve for the moments (-> second derivatives)
        M.solve(moments, d);

        // convert the moments to slopes at the sample points
        this->setSlopesFromMoments_(slopeVec_, moments);
    }

    /*!
     * \brief Create a periodic spline from the already set sampling points.
     *
     * This creates a temporary matrix and right hand side vector.
     */
    void makePeriodicSpline_()
    {
        Matrix M(numSamples() - 1);
        Vector d(numSamples() - 1);
        Vector moments(numSamples() - 1);

        // create linear system of equations. This is a bit hacky,
        // because it assumes that std::vector internally stores its
        // data as a big C-style array, but it saves us from yet
        // another copy operation
        this->makePeriodicSystem_(M, d);

        // solve for the moments (-> second derivatives)
        M.solve(moments, d);

        moments.resize(numSamples());
        for (int i = numSamples() - 2; i >= 0; --i)
            moments[i+1] = moments[i];
        moments[0] = moments[numSamples() - 1];

        // convert the moments to slopes at the sample points
        this->setSlopesFromMoments_(slopeVec_, moments);
    }

    /*!
     * \brief Set the sampling point vectors.
     *
     * This takes care that the order of the x-values is ascending,
     * although the input must be ordered already!
     */
    template <class DestVector, class SourceVector>
    void assignSamplingPoints_(DestVector &destX,
                               DestVector &destY,
                               const SourceVector &srcX,
                               const SourceVector &srcY,
                               int numSamples)
    {
        assert(numSamples >= 2);

        // copy sample points, make sure that the first x value is
        // smaller than the last one
        for (int i = 0; i < numSamples; ++i) {
            int idx = i;
            if (srcX[0] > srcX[numSamples - 1])
                idx = numSamples - i - 1;
            destX[i] = srcX[idx];
            destY[i] = srcY[idx];
        }
    }

    template <class DestVector, class ListIterator>
    void assignFromArrayList_(DestVector &destX,
                              DestVector &destY,
                              const ListIterator &srcBegin,
                              const ListIterator &srcEnd,
                              int numSamples)
    {
        assert(numSamples >= 2);

        // find out wether the x values are in reverse order
        ListIterator it = srcBegin;
        ++it;
        bool reverse = false;
        if ((*srcBegin)[0] > (*it)[0])
            reverse = true;
        --it;

        // loop over all sampling points
        for (int i = 0; it != srcEnd; ++i, ++it) {
            int idx = i;
            if (reverse)
                idx = numSamples - i - 1;
            destX[i] = (*it)[0];
            destY[i] = (*it)[1];
        }
    }

    /*!
     * \brief Set the sampling points.
     *
     * Here we assume that the elements of the source vector have an
     * [] operator where v[0] is the x value and v[1] is the y value
     * if the sampling point.
     */
    template <class DestVector, class ListIterator>
    void assignFromTupleList_(DestVector &destX,
                              DestVector &destY,
                              ListIterator srcBegin,
                              ListIterator srcEnd,
                              int numSamples)
    {
        assert(numSamples >= 2);

        // copy sample points, make sure that the first x value is
        // smaller than the last one

        // find out wether the x values are in reverse order
        ListIterator it = srcBegin;
        ++it;
        bool reverse = false;
        if (std::get<0>(*srcBegin) > std::get<0>(*it))
            reverse = true;
        --it;

        // loop over all sampling points
        for (int i = 0; it != srcEnd; ++i, ++it) {
            int idx = i;
            if (reverse)
                idx = numSamples - i - 1;
            destX[i] = std::get<0>(*it);
            destY[i] = std::get<1>(*it);
        }
    }

    /*!
     * \brief Make the linear system of equations Mx = d which results
     *        in the moments of the full spline.
     */
    template <class Vector, class Matrix>
    void makeFullSystem_(Matrix &M, Vector &d, Scalar m0, Scalar m1)
    {
        makeNaturalSystem_(M, d);

        int n = numSamples() - 1;
        // first row
        M[0][1] = 1;
        d[0] = 6/h_(1) * ( (y_(1) - y_(0))/h_(1) - m0);

        // last row
        M[n][n - 1] = 1;

        // right hand side
        d[n] =
            6/h_(n)
            *
            (m1 - (y_(n) - y_(n - 1))/h_(n));
    }

    /*!
     * \brief Make the linear system of equations Mx = d which results
     *        in the moments of the natural spline.
     */
    template <class Vector, class Matrix>
    void makeNaturalSystem_(Matrix &M, Vector &d)
    {
        M = 0.0;

        // See: J. Stoer: "Numerische Mathematik 1", 9th edition,
        // Springer, 2005, p. 111
        const int n = numSamples() - 1;

        // second to next to last rows
        for (int i = 1; i < n; ++i) {
            Scalar lambda_i = h_(i + 1) / (h_(i) + h_(i + 1));
            Scalar mu_i = 1 - lambda_i;
            Scalar d_i =
                6 / (h_(i) + h_(i + 1))
                *
                ( (y_(i + 1) - y_(i))/h_(i + 1) - (y_(i) - y_(i - 1))/h_(i));

            M[i][i-1] = mu_i;
            M[i][i] = 2;
            M[i][i + 1] = lambda_i;
            d[i] = d_i;
        };

        // See Stroer, equation (2.5.2.7)
        Scalar lambda_0 = 0;
        Scalar d_0 = 0;

        Scalar mu_n = 0;
        Scalar d_n = 0;

        // first row
        M[0][0] = 2;
        M[0][1] = lambda_0;
        d[0] = d_0;

        // last row
        M[n][n-1] = mu_n;
        M[n][n] = 2;
        d[n] = d_n;
    }

    /*!
     * \brief Make the linear system of equations Mx = d which results
     *        in the moments of the periodic spline.
     */
    template <class Matrix, class Vector>
    void makePeriodicSystem_(Matrix &M, Vector &d)
    {
        M = 0.0;

        // See: J. Stoer: "Numerische Mathematik 1", 9th edition,
        // Springer, 2005, p. 111
        const size_t n = numSamples() - 1;

        assert(M.rows() == n);

        // second to next to last rows
        for (size_t i = 2; i < n; ++i) {
            Scalar lambda_i = h_(i + 1) / (h_(i) + h_(i + 1));
            Scalar mu_i = 1 - lambda_i;
            Scalar d_i =
                6 / (h_(i) + h_(i + 1))
                *
                ( (y_(i + 1) - y_(i))/h_(i + 1) - (y_(i) - y_(i - 1))/h_(i));

            M[i-1][i-2] = mu_i;
            M[i-1][i-1] = 2;
            M[i-1][i] = lambda_i;
            d[i-1] = d_i;
        };

        Scalar lambda_n = h_(1) / (h_(n) + h_(1));
        Scalar lambda_1 = h_(2) / (h_(1) + h_(2));;
        Scalar mu_1 = 1 - lambda_1;
        Scalar mu_n = 1 - lambda_n;

        Scalar d_1 =
            6 / (h_(1) + h_(2))
            *
            ( (y_(2) - y_(1))/h_(2) - (y_(1) - y_(0))/h_(1));
        Scalar d_n =
            6 / (h_(n) + h_(1))
            *
            ( (y_(1) - y_(n))/h_(1) - (y_(n) - y_(n-1))/h_(n));


        // first row
        M[0][0] = 2;
        M[0][1] = lambda_1;
        M[0][n-1] = mu_1;
        d[0] = d_1;

        // last row
        M[n-1][0] = lambda_n;
        M[n-1][n-2] = mu_n;
        M[n-1][n-1] = 2;
        d[n-1] = d_n;
    }

    /*!
     * \brief Create a monotonic spline from the already set sampling points.
     *
     * This code is inspired by opm-core's "MonotCubicInterpolator"
     * class and also uses the approach by Fritsch and Carlson, see
     *
     * http://en.wikipedia.org/wiki/Monotone_cubic_interpolation
     */
    template <class Vector>
    void makeMonotonicSpline_(Vector &slopes)
    {
        auto n = numSamples();

        // calculate the slopes of the secant lines
        std::vector<Scalar> delta(n);
        for (int k = 0; k < n - 1; ++k) {
            delta[k] = (y_(k + 1) - y_(k))/(x_(k + 1) - x_(k));
        }

        // calculate the "raw" slopes at the sample points
        for (int k = 1; k < n - 1; ++k)
            slopes[k] = (delta[k - 1] + delta[k])/2;
        slopes[0] = delta[0];
        slopes[n - 1] = delta[n - 2];

        // post-process the "raw" slopes at the sample points
        for (int k = 0; k < n - 1; ++k) {
            if (std::abs(delta[k]) < 1e-50) {
                // make the spline flat if the inputs are equal
                slopes[k] = 0;
                slopes[k + 1] = 0;
                ++ k;
                continue;
            }
            else {
                Scalar alpha = slopes[k] / delta[k];
                Scalar beta = slopes[k + 1] / delta[k];

                if (alpha < 0 || (k > 0 && slopes[k] / delta[k - 1] < 0)) {
                    slopes[k] = 0;
                }
                // limit (alpha, beta) to a circle of radius 3
                else if (alpha*alpha + beta*beta > 3*3) {
                    Scalar tau = 3.0/std::sqrt(alpha*alpha + beta*beta);
                    slopes[k] = tau*alpha*delta[k];
                    slopes[k + 1] = tau*beta*delta[k];
                }
            }
        }
    }

    /*!
     * \brief Convert the moments at the sample points to slopes.
     *
     * This requires to use cubic Hermite interpolation, but it is
     * required because for monotonic splines the second derivative is
     * not continuous.
     */
    template <class MomentsVector, class SlopeVector>
    void setSlopesFromMoments_(SlopeVector &slopes, const MomentsVector &moments)
    {
        int n = numSamples();

        // evaluate slope at the rightmost point.
        // See: J. Stoer: "Numerische Mathematik 1", 9th edition,
        // Springer, 2005, p. 109
        Scalar mRight;

        {
            Scalar h = this->h_(n - 1);
            Scalar x = h;
            //Scalar x_1 = 0;

            Scalar A =
                (y_(n - 1) - y_(n - 2))/h
                -
                h/6*(moments[n-1] - moments[n - 2]);

            mRight =
                //- moments[n - 2] * x_1*x_1 / (2 * h)
                //+
                moments[n - 1] * x*x / (2 * h)
                +
                A;
        }

        // evaluate the slope for the first n-1 sample points
        for (int i = 0; i < n - 1; ++ i) {
            // See: J. Stoer: "Numerische Mathematik 1", 9th edition,
            // Springer, 2005, p. 109
            Scalar h_i = this->h_(i + 1);
            //Scalar x_i = 0;
            Scalar x_i1 = h_i;

            Scalar A_i =
                (y_(i+1) - y_(i))/h_i
                -
                h_i/6*(moments[i+1] - moments[i]);

            slopes[i] =
                - moments[i] * x_i1*x_i1 / (2 * h_i)
                +
                //moments[i + 1] * x_i*x_i / (2 * h_i)
                //+
                A_i;

        }
        slopes[n - 1] = mRight;
    }


    // evaluate the spline at a given the position and given the
    // segment index
    Scalar eval_(Scalar x, int i) const
    {
        // See http://en.wikipedia.org/wiki/Cubic_Hermite_spline
        Scalar delta = h_(i + 1);
        Scalar t = (x - x_(i))/delta;

        return
            h00_(t) * y_(i)
            + h10_(t) * slope_(i)*delta
            + h01_(t) * y_(i + 1)
            + h11_(t) * slope_(i + 1)*delta;
    }

    // evaluate the derivative of a spline given the actual position
    // and the segment index
    Scalar evalDerivative_(Scalar x, int i) const
    {
        // See http://en.wikipedia.org/wiki/Cubic_Hermite_spline
        Scalar delta = h_(i + 1);
        Scalar t = (x - x_(i))/delta;
        Scalar alpha = 1 / delta;

        return
            alpha *
            (h00_prime_(t) * y_(i)
             + h10_prime_(t) * slope_(i)*delta
             + h01_prime_(t) * y_(i + 1)
             + h11_prime_(t) * slope_(i + 1)*delta);
    }

    // evaluate the second derivative of a spline given the actual
    // position and the segment index
    Scalar evalDerivative2_(Scalar x, int i) const
    {
        // See http://en.wikipedia.org/wiki/Cubic_Hermite_spline
        Scalar delta = h_(i + 1);
        Scalar t = (x - x_(i))/delta;
        Scalar alpha = 1 / delta;

        return
            alpha*alpha
            *(h00_prime2_(t) * y_(i)
              + h10_prime2_(t) * slope_(i)*delta
              + h01_prime2_(t) * y_(i + 1)
              + h11_prime2_(t) * slope_(i + 1)*delta);
    }

    // evaluate the third derivative of a spline given the actual
    // position and the segment index
    Scalar evalDerivative3_(Scalar x, int i) const
    {
        // See http://en.wikipedia.org/wiki/Cubic_Hermite_spline
        Scalar delta = h_(i + 1);
        Scalar t = (x - x_(i))/delta;
        Scalar alpha = 1 / delta;

        return
            alpha*alpha*alpha
            *(h00_prime3_(t)*y_(i)
              + h10_prime3_(t)*slope_(i)*delta
              + h01_prime3_(t)*y_(i + 1)
              + h11_prime3_(t)*slope_(i + 1)*delta);
    }

    // hermite basis functions
    Scalar h00_(Scalar t) const
    { return (2*t - 3)*t*t + 1; }

    Scalar h10_(Scalar t) const
    { return ((t - 2)*t + 1)*t; }

    Scalar h01_(Scalar t) const
    { return (-2*t + 3)*t*t; }

    Scalar h11_(Scalar t) const
    { return (t - 1)*t*t; }

    // first derivative of the hermite basis functions
    Scalar h00_prime_(Scalar t) const
    { return (3*2*t - 2*3)*t; }

    Scalar h10_prime_(Scalar t) const
    { return (3*t - 2*2)*t + 1; }

    Scalar h01_prime_(Scalar t) const
    { return (-3*2*t + 2*3)*t; }

    Scalar h11_prime_(Scalar t) const
    { return (3*t - 2)*t; }

    // second derivative of the hermite basis functions
    Scalar h00_prime2_(Scalar t) const
    { return 2*3*2*t - 2*3; }

    Scalar h10_prime2_(Scalar t) const
    { return 2*3*t - 2*2; }

    Scalar h01_prime2_(Scalar t) const
    { return -2*3*2*t + 2*3; }

    Scalar h11_prime2_(Scalar t) const
    { return 2*3*t - 2; }

    // third derivative of the hermite basis functions
    Scalar h00_prime3_(Scalar /*t*/) const
    { return 2*3*2; }

    Scalar h10_prime3_(Scalar /*t*/) const
    { return 2*3; }

    Scalar h01_prime3_(Scalar /*t*/) const
    { return -2*3*2; }

    Scalar h11_prime3_(Scalar /*t*/) const
    { return 2*3; }

    // returns the monotonicality of an interval of a spline segment
    //
    // The return value have the following meaning:
    //
    // 3: spline is constant within interval [x0, x1]
    // 1: spline is monotonously increasing in the specified interval
    // 0: spline is not monotonic (or constant) in the specified interval
    // -1: spline is monotonously decreasing in the specified interval
    int monotonic_(int i, Scalar x0, Scalar x1, int &r) const
    {
        // coefficients of derivative in monomial basis
        Scalar a = 3*a_(i);
        Scalar b = 2*b_(i);
        Scalar c = c_(i);

        if (std::abs(a) < 1e-20 && std::abs(b) < 1e-20 && std::abs(c) < 1e-20)
            return 3; // constant in interval, r stays unchanged!

        Scalar disc = b*b - 4*a*c;
        if (disc < 0) {
            // discriminant of derivative is smaller than 0, i.e. the
            // segment's derivative does not exhibit any extrema.
            if (x0*(x0*a + b) + c > 0) {
                r = (r==3 || r == 1)?1:0;
                return 1;
            }
            else {
                r = (r==3 || r == -1)?-1:0;
                return -1;
            }
        }
        disc = std::sqrt(disc);
        Scalar xE1 = (-b + disc)/(2*a);
        Scalar xE2 = (-b - disc)/(2*a);

        if (disc == 0) {
            // saddle point -> no extrema
            if (xE1 == x0)
                // make sure that we're not picking the saddle point
                // to determine whether we're monotonically increasing
                // or decreasing
                x0 = x1;
            if (x0*(x0*a + b) + c > 0) {
                r = (r==3 || r == 1)?1:0;
                return 1;
            }
            else {
                r = (r==3 || r == -1)?-1:0;
                return -1;
            }
        };
        if ((x0 < xE1 && xE1 < x1) ||
            (x0 < xE2 && xE2 < x1))
        {
            // there is an extremum in the range (x0, x1)
            r = 0;
            return 0;
        }
        // no extremum in range (x0, x1)
        x0 = (x0 + x1)/2; // pick point in the middle of the interval
                          // to avoid extrema on the boundaries
        if (x0*(x0*a + b) + c > 0) {
            r = (r==3 || r == 1)?1:0;
            return 1;
        }
        else {
            r = (r==3 || r == -1)?-1:0;
            return -1;
        }
    }

    /*!
     * \brief Find all the intersections of a segment of the spline
     *        with a cubic polynomial within a specified interval.
     */
    int intersectSegment_(Scalar *sol,
                          int segIdx,
                          Scalar a, Scalar b, Scalar c, Scalar d,
                          Scalar x0 = -1e100, Scalar x1 = 1e100) const
    {
        int n = Opm::invertCubicPolynomial(sol,
                                           a_(segIdx) - a,
                                           b_(segIdx) - b,
                                           c_(segIdx) - c,
                                           d_(segIdx) - d);
        x0 = std::max(x_(segIdx), x0);
        x1 = std::min(x_(segIdx+1), x1);

        // filter the intersections outside of the specified interval
        int k = 0;
        for (int j = 0; j < n; ++j) {
            if (x0 <= sol[j] && sol[j] <= x1) {
                sol[k] = sol[j];
                ++k;
            }
        }
        return k;
    }

    // find the segment index for a given x coordinate
    int segmentIdx_(Scalar x) const
    {
        // bisection
        int iLow = 0;
        int iHigh = numSamples() - 1;

        while (iLow + 1 < iHigh) {
            int i = (iLow + iHigh) / 2;
            if (x_(i) > x)
                iHigh = i;
            else
                iLow = i;
        };
        return iLow;
    }

    /*!
     * \brief Returns x[i] - x[i - 1]
     */
    Scalar h_(int i) const
    {
        assert(x_(i) > x_(i-1)); // the sampling points must be given
                                 // in ascending order
        return x_(i) - x_(i - 1);
    }

    /*!
     * \brief Returns the y coordinate of the i-th sampling point.
     */
    Scalar x_(int i) const
    { return xPos_[i]; }

    /*!
     * \brief Returns the y coordinate of the i-th sampling point.
     */
    Scalar y_(int i) const
    { return yPos_[i]; }

    /*!
     * \brief Returns the slope (i.e. first derivative) of the spline at
     *        the i-th sampling point.
     */
    Scalar slope_(int i) const
    { return slopeVec_[i]; }

    // returns the coefficient in front of the x^3 term. In Stoer this
    // is delta.
    Scalar a_(int i) const
    { return evalDerivative3_(/*x=*/0, i)/6.0; }

    // returns the coefficient in front of the x^2 term In Stoer this
    // is gamma.
    Scalar b_(int i) const
    { return evalDerivative2_(/*x=*/0, i)/2.0; }

    // returns the coefficient in front of the x^1 term. In Stoer this
    // is beta.
    Scalar c_(int i) const
    { return evalDerivative_(/*x=*/0, i); }

    // returns the coefficient in front of the x^0 term. In Stoer this
    // is alpha.
    Scalar d_(int i) const
    { return eval_(/*x=*/0, i); }

    Vector xPos_;
    Vector yPos_;
    Vector slopeVec_;
};
}

#endif
